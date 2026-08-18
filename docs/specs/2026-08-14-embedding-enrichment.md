# SPEC: embedding enrichment + `semantic_search`

Status: draft (pre-implementation)
Date: 2026-08-14
Branch: `spec/embedding-enrichment`

## Problem

graphify's search is lexical. `serve._score_nodes` matches query terms against
`norm_label`, `label_tokens`, `source_file`, and the node id, gated by a trigram
candidate index (`serve._node_search_text`, `serve._get_trigram_index`). That is
fast and deterministic, and it fails on the query class agents ask most:

> "where do we check that a token hasn't expired?"

when the code is `def _assert_not_stale(claims)` in `auth/jwt.py`. No trigram of
the query appears in any indexed field, so the node is not even a candidate. The
existing `semantically_similar_to` edge relation (`llm.py:479`,
`analyze.py:261`) is populated only when the extraction LLM happens to assert
it — there is no vector signal anywhere in the codebase today (confirmed: no
`embed.py`, no `EMBEDDING_BACKENDS`, zero hits for `embedding` outside
unrelated "escape for embedding in YAML" docstrings).

This spec adds embeddings as a **second, additive** retrieval signal, and
exposes it three ways: a new `semantic_search` MCP tool, a blend into
`query_graph`'s existing scoring, and vector properties + a native vector index
in Neo4j for users who push their graph there.

## Goals

- Retrieve nodes by meaning, not substring, for both code and prose nodes.
- Cost-free and offline by default — local Ollama via the Bifrost gateway.
- Fully incremental: re-embedding a 5k-node graph after a 3-file edit must cost
  3 files of embedding calls, not 5k.
- Additive and reversible: a graph with no embeddings behaves exactly as today,
  and every new code path is a no-op when the sidecar is absent.
- Neo4j push carries the vectors and creates a queryable vector index, so Aura
  is a first-class consumer, not just graphify's own MCP server.

## Non-goals

- Replacing lexical search. Embeddings *blend into* it (see §6).
- Re-ranking with an LLM, HyDE, or query expansion.
- An ANN index (HNSW/IVF) inside graphify. See §4 for why brute force is correct
  at graphify's scale.
- Embedding hyperedges or edges. Nodes only in this spec.
- Chunking a node's body across multiple vectors. One vector per node (§3).

## 1. Scope of what gets embedded

graphify has six `file_type` values (`validate.py:4`):
`code`, `document`, `paper`, `image`, `rationale`, `concept`.

| `file_type` | Embedded? | Model | Rationale |
| --- | --- | --- | --- |
| `code` | yes | **code model** | identifiers + signature are code-shaped |
| `document`, `paper` | yes | **text model** | prose |
| `rationale`, `concept` | yes | **text model** | LLM-authored prose |
| `image` | no | — | no text to embed; vision embedding is out of scope |

**Two models, not one.** A code-specialized embedder underperforms on prose and
vice versa. This is the single most structurally important requirement in this
spec: it forces two model slots, two cache namespaces, and two Neo4j vector
indexes all the way through. A design that assumes one model will not extend.

Defaults: code → `nomic-embed-code`, text → `nomic-embed-text`.

**Vectors from different models are not comparable.** Cosine between a
`nomic-embed-code` vector and a `nomic-embed-text` vector is meaningless. This
partitions the whole system into two independent vector spaces, which drives
§5's per-space search and §7's two indexes.

## 2. Node text construction

`source_location` is a **start line only** — `f"L{node.start_point[0] + 1}"`
(`extract.py:332` and ~20 identical sites). There is no end line, so a node's
body extent is unknown and cannot be read back from disk. Embedding "the
function body" is therefore not available without an extractor change, which
this spec deliberately does not make.

What is available per node: `label`, `id`, `source_file`, `file_type`,
`rationale` (often `None` for AST-extracted code), and community membership.

### 2.1 Code nodes

Compose a synthetic document from available signal:

```
{label}
{qualified path form of source_file}
{rationale if present}
{space-joined labels of up to 10 highest-degree neighbours}
```

The neighbour labels are the load-bearing part. A bare label
(`_assert_not_stale`) is a weak embedding target; the same label surrounded by
`verify_token`, `JWTClaims`, `TokenExpiredError` carries real semantics. This
uses only graph structure already in memory — no file reads, no LLM calls.

Cap at 512 tokens (~2000 chars); truncate neighbours first, never the label.

### 2.2 Text nodes

Prefer `rationale` (the LLM's own description) as the primary text, per the
rationale-first sourcing decision. Fall back to `label` + `source_file` when
`rationale` is absent. Same neighbour augmentation, same cap.

### 2.3 Determinism

Text construction must be a pure function of the node's graph data, with
neighbours sorted by `(-degree, id)` so the tie-break is total. Two runs over an
unchanged graph must produce byte-identical text, or the cache never hits.

## 3. Storage: sidecar + Neo4j, not `graph.json`

Embeddings live in **`graphify-out/embeddings.npz`**, not in `graph.json`.

`graph.json` is written via `json_graph.node_link_data` (`export.py:302`), is
diffed by users, is loaded whole into memory by ~10 call sites, and is guarded
by a 512 MiB cap (`security.py:32`). Inlining 5k × 768 float64 as JSON text adds
roughly 75 MB of unreadable float noise to an artifact whose readability is a
feature. It would also break the incremental story: a single re-embedded node
rewrites the whole file.

Sidecar format — `numpy.savez` with one entry per vector space:

```
code_ids     : (N_c,)        unicode array of node ids
code_vecs    : (N_c, 768)    float32, L2-normalized
code_meta    : json string   {model, backend, dim, graphify_version, created_at}
text_ids     : (N_t,)
text_vecs    : (N_t, 768)
text_meta    : json string
```

`float32` not `float64`: halves the file for no measurable recall change, and
cosine is computed in float32 anyway.

**Vectors are stored pre-normalized.** Ollama's OpenAI-compatible
`/v1/embeddings` already returns unit vectors (verified: norm = 1.000000, dim =
768 for `nomic-embed-text`), but the native `/api/embeddings` route does not.
Normalizing on write makes cosine a plain dot product and makes the invariant
hold regardless of which route a future backend uses. `enrich_embeddings` must
re-normalize defensively rather than trust the provider.

`.npz` must be added to `export._BACKUP_ARTIFACTS` (`export.py:24`) alongside
`graph.json` — it is not regenerable without re-running the embedding pass, which
is exactly the criterion that list documents.

### 3.1 Neo4j is the second storage tier

The sidecar is graphify's own store. **Neo4j gets the vectors too**, as node
properties plus a native vector index — see §7. Both tiers are populated from
the same `enrich_embeddings` output, so they cannot drift.

## 4. Search: brute force, and why

`numpy>=1.21` is a **core** dependency (`pyproject.toml`), not optional. So
brute-force cosine needs no new dependency and no build step.

For a 5k-node graph: `vecs @ q` is a 5000×768 float32 matmul ≈ 4 MB of data,
which lands in ~2–4 ms on any modern CPU — well inside MCP call overhead. An
ANN index would add a dependency, a build step, a serialization format, and
recall error, to optimize something that is already imperceptible. At 500k nodes
this decision would flip; graphify graphs are not that size, and the 512 MiB
`graph.json` cap bounds them well below it.

Vector loading must be **lazy and cached on the graph object**, mirroring
`_get_trigram_index`'s pattern (`serve.py:350`) — a server that never receives a
`semantic_search` call must never touch the `.npz`.

## 5. Query-side model routing

A query is one string, but there are two vector spaces. Embedding the query with
both models and searching both is the only correct option: the user's query does
not announce whether they want code or prose, and results from the two spaces
are merged by score after each is searched in its own space.

This means **two embedding calls per query**. Both are local, ~15 ms each, and
must be issued concurrently. A query-embedding cache (LRU, in-process, keyed by
`(model, text)`) makes repeat queries free within a server session.

Scores from different spaces are not calibrated against each other. Merge by
taking the per-space top-k and interleaving by raw cosine, and **label each
result with its space** in the output so a caller can see which signal produced
it. Do not claim cross-space score comparability.

## 6. Integration surfaces

### 6.1 `semantic_search` MCP tool (new)

```json
{
  "name": "semantic_search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query":       {"type": "string"},
      "top_k":       {"type": "integer", "default": 10},
      "file_type":   {"type": "array", "items": {"type": "string"},
                      "description": "Optional: restrict to these file_types"},
      "min_score":   {"type": "number", "default": 0.3}
    },
    "required": ["query"]
  }
}
```

Registered in `serve.list_tools`'s `_tools` list (`serve.py:1519`) and in
`_handlers` for `call_tool` dispatch (`serve.py:1978`). The `project_path`
injection loop (`serve.py:1653`) picks it up automatically — no change needed
there.

Output must route every LLM-derived field through `sanitize_label`, exactly as
`_tool_get_node` does (`serve.py:1707`) — embeddings do not change the fact that
node labels are untrusted LLM output.

**Absent-sidecar behaviour**: return a plain instruction to run
`graphify embed .`, not an error. `call_tool` wraps exceptions into
`"Error executing ..."` text (`serve.py:1984`), which reads to an agent as a
malfunction rather than a missing optional step.

### 6.2 `query_graph` blend (existing tool, additive)

`_score_query` currently produces lexical scores. Add semantic scores as an
extra contribution to seed selection, so `query_graph`'s existing BFS/DFS
behaviour improves without a new tool call. Gated: when no sidecar exists, the
blend weight is zero and scoring is bit-identical to today.

The blend must not let a semantic hit *displace* a strong lexical hit — an exact
label match is still the best answer for "show me `validate_input`". Lexical
score keeps priority; semantic fills the tail.

### 6.3 CLI

- `graphify embed .` — standalone re-embed of an existing graph.
- `graphify extract . --embed` — embed as part of extraction.

Note the CLI is **hand-rolled**, not argparse subparsers: `__main__.py` reads
`sys.argv[1]` and calls `cli.dispatch_command` (`__main__.py:705-712`), with
help text as literal `print` calls. `embed` must be added to `dispatch_command`,
to the `--help` block, and must not be added to `_FREE_TEXT_CMDS`
(`__main__.py:705`) so the help guard applies.

## 7. Neo4j: vector properties + native vector index

`push_to_neo4j` (`graphdb.py:9`) today filters node properties to scalars:

```python
props = {
    k: v for k, v in data.items()
    if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
}
```

`isinstance(v, (str, int, float, bool))` **silently drops any list**, so an
embedding would vanish with no warning. This is the prop-filter fix this spec
depends on: allow a `list` whose members are all `int`/`float` (Neo4j stores
that as a `LIST<FLOAT>`, which is what its vector index requires), while
continuing to drop lists of dicts and other non-primitive nesting. The same
filter appears twice more — the edge loop (`graphdb.py:64`) and
`push_to_falkordb` (`graphdb.py:145`, `:162`) — so the fix belongs in one shared
helper applied at all four sites.

`bool` must be checked before `int` in any numeric test, since `bool` is a
subclass of `int` in Python.

### 7.1 Index creation

There is currently **no vector-index or `CREATE INDEX` support anywhere** in
graphify (grep for `vector`/`CREATE INDEX` in `graphdb.py`/`export.py`: zero
hits). This spec adds it:

```cypher
CREATE VECTOR INDEX graphify_code_embeddings IF NOT EXISTS
FOR (n:Code) ON (n.embedding_code)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
```

Two indexes, one per vector space, matching §1's two-model split. Property names
`embedding_code` / `embedding_text` keep the spaces distinct on the node so a
Cypher query cannot accidentally compare across them.

Node labels come from `_safe_label(data.get("file_type", ...).capitalize())`
(`graphdb.py:54`), so the label set is `Code`, `Document`, `Paper`, `Rationale`,
`Concept`, `Image` — the text index must cover the four prose labels, which
means either one index per label or a shared label added at push time. **Adding
a shared `:Embedded` label to every vector-carrying node is the simpler
option** and is what this spec specifies: one code index on `:Embedded`, one
text index on `:Embedded`, distinguished by property.

Index creation is idempotent (`IF NOT EXISTS`) and must be skipped gracefully
when the target is not Neo4j 5.13+ or is FalkorDB, whose vector support differs.
A failed index creation must **not** fail the push — the nodes and their vectors
are still correctly written, and the index is a queryability improvement.

Acceptance for this section is explicitly *not* "the push succeeded". It is: run
a `db.index.vector.queryNodes` call against Aura and get sensible neighbours
back. Vectors that round-trip but are not queryable are a silent failure.

## 8. Cache design

Two namespaces, keyed by `(backend, model)`, under `graphify-out/cache/`
alongside the existing `ast/` and `semantic/` kinds (`cache.cache_dir`,
`cache.py:839`). Switching the code model must not invalidate text vectors.

Cache key per node: `sha256` of the **constructed text** from §2, not the source
file hash. A node's embedding text depends on its neighbours' labels, so an edit
to file A changes the embedding text of a node in file B. Keying on the source
hash would serve a stale vector; keying on the constructed text is correct by
construction and makes the incremental goal fall out for free — unchanged text
means a hit, no call.

Follow the existing cache's disciplines rather than inventing new ones:
corrupt-entry counting (`cache.py:926`), atomic writes, and a prune pass
mirroring `prune_semantic_cache` (`cache.py:1073`) so vectors for deleted nodes
do not accumulate forever.

## 9. Backend wiring

Add `EMBEDDING_BACKENDS` to `llm.py`, parallel to the existing `BACKENDS` dict
(`llm.py:100`) but separate — embedding endpoints take `input`, return `data[].embedding`,
and have no `temperature`/`max_tokens`/`vision`/system-prompt concepts. Reusing
`BACKENDS` would mean threading "is this a chat or an embedding model" through
every consumer of that dict.

Backends: `ollama` (default), `openai`. Explicitly excluded — `claude-cli`
(no embedding endpoint), `bedrock` (different API shape), `claude`/`anthropic`
(Anthropic ships no embedding model). Requesting an excluded backend must fail
with a message naming the supported set, not fall through to a 404.

`_call_embeddings` mirrors `_call_openai_compat`'s hardening (`llm.py:1165`),
which is battle-tested against exactly this local-Ollama topology:

- `GRAPHIFY_API_TIMEOUT`, default 600s (`_resolve_api_timeout`).
- **Zero SDK retries for ollama** unless `GRAPHIFY_MAX_RETRIES` is set
  explicitly — a local server does not rate-limit, and 6 retries turn a timeout
  into a ~21-minute block (`llm.py:1198-1200`, issue #1686).
- Batch inputs per call, with a documented batch size, since embedding one node
  per HTTP round-trip dominates wall-clock on a 5k-node graph.
- Validate `dim` against the sidecar's recorded `meta.dim` on every load and
  refuse to mix dimensions — a silent model swap otherwise corrupts the space.

`OPENAI_BASE_URL` (or its embedding-specific equivalent) points at the Bifrost
gateway, so this reaches local Ollama through the same route the rest of the
stack uses. Bifrost passes Ollama's normalized embeddings through losslessly
(measured L1 diff 0.000000), so gateway-vs-direct is not a correctness variable.

## 10. Acceptance criteria (test-first)

These tests are written **before** the implementation they describe. Each
increment in §11 lands with its tests passing.

### `tests/test_embed_text.py` — §2
- Code-node text includes label, path, and neighbour labels.
- Neighbour ordering is `(-degree, id)`; a degree tie is broken by id, and the
  output is byte-identical across two runs.
- Cap truncates neighbours, never the label.
- Text nodes prefer `rationale`; fall back to `label` when it is `None`.
- `image` nodes are excluded from both spaces.

### `tests/test_embed_backends.py` — §9
- `EMBEDDING_BACKENDS` exposes `ollama` and `openai`.
- `claude-cli`, `bedrock`, `claude` raise a message naming the supported set.
- Ollama default is 0 SDK retries; `GRAPHIFY_MAX_RETRIES=3` overrides it.
- A returned vector of the wrong dimension raises rather than being stored.
- Batching: 250 nodes at batch size 100 issues exactly 3 calls.
- Un-normalized provider output is normalized on write (feed a mock returning
  norm-5 vectors; assert stored norm is 1.0).

### `tests/test_embed_cache.py` — §8
- Code and text namespaces are independent: changing the code model leaves text
  entries hittable.
- Cache key follows constructed text, not file hash: **editing file A must
  invalidate the entry for a node in file B whose neighbour label changed.**
  This is the test that catches the tempting-but-wrong source-hash key.
- Re-running `embed` on an unchanged graph issues **zero** embedding calls.
- Corrupt `.npz` is a miss, is counted, and does not raise.
- Prune drops vectors for nodes no longer in the graph.

### `tests/test_embed_search.py` — §4, §5
- Cosine top-k ordering is correct against a hand-built vector fixture.
- `min_score` filters; `file_type` filters.
- Results carry their vector space, and no code-vs-text score comparison is
  asserted.
- Query-embedding LRU: the same query twice issues one embedding call per space.
- Lazy load: constructing the server with no `semantic_search` call never opens
  the `.npz`.

### `tests/test_embed_serve.py` — §6.1, §6.2
- `semantic_search` appears in `list_tools` and gets `project_path` injected by
  the existing loop.
- Missing sidecar returns the run-`graphify embed` instruction, **not** an
  `"Error executing"` string.
- Labels are `sanitize_label`d — feed a node label containing an injection
  sentinel and assert it is neutralized in the output.
- `query_graph` on a graph with no sidecar produces **bit-identical** output to
  the pre-change implementation (the no-regression guard).
- An exact-label lexical hit still ranks first when a semantic near-miss scores
  higher.

### `tests/test_embed_graphdb.py` — §7
- A `list[float]` node property survives the prop filter and reaches the driver.
- A `list[dict]` is still dropped.
- `True` is not coerced to `1` by a numeric-list check (bool-before-int).
- All four filter sites (node/edge × neo4j/falkordb) use the shared helper.
- `CREATE VECTOR INDEX` is emitted once per space, is `IF NOT EXISTS`, and a
  raised exception from index creation does **not** fail the push.
- Round-trip against a mock driver: pushed vector equals sidecar vector.

### `tests/test_embed_cli.py` — §6.3
- `graphify embed .` dispatches; `graphify embed --help` shows help rather than
  running (the `_FREE_TEXT_CMDS` guard).
- `extract --embed` runs embedding after the graph is written.
- Both are no-ops with a clear message when the graph has zero nodes.

### Manual acceptance (not unit-testable)
Against `neo4j/neo4j-graphrag-python` pushed to Aura: a
`db.index.vector.queryNodes` call returns sensible neighbours for a natural
language query. Per §7.1, a successful push alone does not count.

## 11. Build order

Each step's tests pass before the next begins.

1. **`graphdb.py`** — shared prop-filter helper at all four sites +
   `CREATE VECTOR INDEX`. First because it is independent of everything else and
   is currently a silent data-loss bug.
2. **`llm.py`** — `EMBEDDING_BACKENDS`, `_call_embeddings`.
3. **`cache.py`** — two-namespace embedding cache + prune.
4. **`embed.py`** (new) — text construction, `enrich_embeddings`, `.npz` I/O.
5. **`serve.py`** — `semantic_search`, then the `query_graph` blend.
6. **`cli.py` / `__main__.py`** — `graphify embed`, `extract --embed`.

## 12. Open questions

1. **`nomic-embed-code` is not pulled yet** (~14 GB). Only `nomic-embed-text`
   (137M, 768-dim) is present locally. Steps 1–3 do not need it; step 4's code
   path does. If its dimensionality differs from 768, §3's fixed-dim sidecar
   layout still holds (dims are per-space and recorded in `meta`), but the two
   Neo4j indexes will declare different `vector.dimensions` — which the §7.1
   design already allows, since they are separate indexes.
2. **`query_graph` blend weight** is unspecified. Deliberately: it needs
   measurement against a real graph, and §6.2's no-regression test holds the
   gate closed until then. Shipping §6.1 alone is a valid first release.
3. **Vector staleness vs. graph staleness** is out of scope, consistent with the
   existing deferral of graph-staleness checking as a separate subsystem.
