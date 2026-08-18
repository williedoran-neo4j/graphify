# Steel Thread Plan: embedding enrichment + `semantic_search`

Source spec: `/Users/williedoran/Dev/graphify-fork/docs/specs/2026-08-14-embedding-enrichment.md`
This file is the authoritative spec for the work. §3 Contracts and §7 Invariants
are cross-cutting — read them on **every** requirement, not just the first.

---

## 1. Steel Thread Blueprint

The one transaction a user (agent) would pay for: **ask a graph a meaning-based
question and get back the right nodes** — the query class lexical search fails
on (`"where do we check a token hasn't expired?"` → `_assert_not_stale`). The
spec itself confirms this is the shippable core: *"Shipping §6.1 alone is a valid
first release."* Everything else (second model, Neo4j tier, cache, blend, CLI) is
depth layered onto this shape.

The thinnest end-to-end path, one **text vector space** only:

1. **Enrich.** A caller invokes `enrich_embeddings(graph)` on an in-memory
   graphify graph. For each embeddable **text-family** node
   (`document`/`paper`/`rationale`/`concept`), it builds a deterministic text
   string, gets a vector for it, **L2-normalizes on write**, and writes a sidecar
   `graphify-out/embeddings.npz` holding `text_ids`, `text_vecs` (float32,
   unit-norm), and `text_meta`. `image` nodes are skipped.
2. **Serve.** The MCP server exposes `semantic_search`. A call
   `semantic_search(query=…)` **lazily** loads the `.npz` (cached on the graph
   object, never touched until the first call), embeds the query, computes
   brute-force cosine (`vecs @ q`) in numpy, takes top-k, drops rows below
   `min_score`, optionally filters by `file_type`, **routes every label through
   `sanitize_label`**, tags each result with its vector `space`, and returns the
   ranked list as MCP text. If the sidecar is absent it returns a plain
   *"run `graphify embed .`"* instruction — never an `"Error executing"` string.

**Why this is the thread and not more.** It cuts vertically through every
boundary the feature owns — node → text → embed → normalize → sidecar on disk →
lazy load → cosine → rank → sanitize → MCP output — so the contracts between
those pieces are discovered now, while each piece is still a stub.

**What is real in the thread, and why (stub-by-necessity):**
- **numpy cosine** — `numpy` is a *core* dependency; brute force needs no new
  code or build step. Real from day one; there is no "make search real" slice.
- **The `.npz` sidecar on disk** — `numpy.savez` is trivial and real; a fake
  in-memory store would be throwaway since disk round-trip is the point of §3.
- **The MCP serve surface** — `serve.list_tools`/`call_tool` are already there;
  registering a tool is real, not stubbed.

**What is hardcoded/stubbed in the thread, and why:**
- **`_call_embeddings` returns a deterministic fixed vector per input.** The
  spec mocks the backend in *every* unit test (`test_embed_backends` feeds a
  mock; `test_embed_search` uses a hand-built fixture); the real signal only
  appears in Manual acceptance. Half the backend is also genuinely unavailable —
  `nomic-embed-code` is not pulled (14 GB, open question #1). A deterministic
  stub keeps the core contract test hermetic and is replaced by the real local
  call in the very first slice (R3), which is low-risk because `nomic-embed-text`
  *is* pulled locally. This is the one place the thread trades a small extra
  slice for a deterministic, offline-runnable contract suite — a deliberate call.
- **Text construction is minimal** — `f"{label}\n{source_file}"`. Neighbour
  augmentation is "the load-bearing part" but it is depth, not shape; deferred to
  R5. Determinism still holds (invariant I2) even for the minimal form.
- **One vector space (text).** The two-model split is "the single most
  structurally important requirement," so the sidecar layout, search, and result
  schema are **space-keyed from day one** (`text_*` entries, `space` label on
  every result) — adding `code_*` in R10 is purely additive, never a rename.

---

## 2. Deferred (out of scope for the thread)

Nothing below is dropped; each has a home in §5/§6/§7 and a backlog id in §8.

- **Real embedding backend** (`EMBEDDING_BACKENDS`, `_call_embeddings`, Bifrost
  gateway routing, excluded-backend errors, retry/timeout policy, dim validation)
  — §9 → R3.
- **Rich text construction** (neighbour augmentation, qualified path,
  rationale-first sourcing, 512-token cap) — §2.1/§2.2 → R5.
- **Second vector space / code model** (`nomic-embed-code`, `code_*` sidecar,
  code-node text, dual-space search + merge/interleave, dual query embedding) —
  §1/§2.1/§5 → R10.
- **Embedding cache** (two namespaces, sha256-of-constructed-text key, atomic
  writes, corrupt-count, prune) and the **incremental** guarantee — §8 → R6.
- **Batching + query-embedding LRU + concurrent dual embed** — §5/§9 → R7.
- **CLI** (`graphify embed .`, `graphify extract . --embed`, `dispatch_command`,
  `--help`, `_FREE_TEXT_CMDS` guard, zero-node no-op) — §6.3 → R8.
- **`query_graph` blend** (semantic scores into `_score_query` seed selection,
  gated, lexical-priority) — §6.2 → R11.
- **Neo4j tier** (shared prop-filter fix at 4 sites, `LIST<FLOAT>` props,
  `:Embedded` label, two `CREATE VECTOR INDEX`, graceful skip, Aura queryNodes
  acceptance) — §7 → R9.
- **`.npz` in `export._BACKUP_ARTIFACTS`** — §3 → R4.

**Permanent non-goals** (never built): replacing lexical search; LLM re-ranking /
HyDE / query expansion; an ANN index (HNSW/IVF) inside graphify; embedding
edges/hyperedges; chunking a node across multiple vectors; vector-vs-graph
staleness reconciliation (open question #3).

---

## 3. Contracts

### C1 — `build_node_text(node) -> str | None`
Pure, deterministic function of the node's in-memory graph data only (no file
reads, no LLM calls).
- **Thread form (text family):** returns `f"{label}\n{source_file}"`.
- `image` nodes → returns `None` (excluded from every space).
- Byte-identical output across two runs over an unchanged graph (invariant I2).
- *Later (R5):* neighbour augmentation, path form, rationale-first, 512-token cap
  — additive to the string, contract signature unchanged.

### C2 — Embedding backend seam `_call_embeddings(backend, model, inputs: list[str]) -> list[list[float]]`
- Returns exactly one vector per input, in order.
- **Thread:** hardcoded deterministic vectors (green step; not real yet).
- Callers **must not** trust provider normalization — the sidecar writer
  re-normalizes (see C3, invariant I3).
- *Later (R3):* real ollama/openai HTTP via Bifrost; excluded backends raise;
  retry/timeout/dim policy per §9.

### C3 — Sidecar `graphify-out/embeddings.npz` (via `numpy.savez`)
Space-keyed; the thread writes only the `text_*` group, but the layout reserves
the `code_*` group so R10 is additive:
```
text_ids  : (N_t,)      unicode array of node ids
text_vecs : (N_t, D)    float32, every row L2-normalized (‖v‖ == 1.0)
text_meta : json string {model, backend, dim, graphify_version, created_at}
# reserved for R10, absent in the thread:
code_ids, code_vecs, code_meta
```
- `D` (dim) is recorded in `meta.dim`; loaders validate incoming vectors against
  it and refuse to mix dimensions (invariant I5).

### C4 — `semantic_search` MCP tool
Input schema (verbatim from §6.1):
```json
{
  "name": "semantic_search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query":     {"type": "string"},
      "top_k":     {"type": "integer", "default": 10},
      "file_type": {"type": "array", "items": {"type": "string"}},
      "min_score": {"type": "number", "default": 0.3}
    },
    "required": ["query"]
  }
}
```
- Registered in `serve.list_tools`'s `_tools` (`serve.py:1519`) and `_handlers`
  (`serve.py:1978`); the `project_path` injection loop (`serve.py:1653`) picks it
  up with no change.
- **Internal result shape** (per node): `{id: str, label: str, score: float,
  space: "text", file_type: str}`. `label` is already `sanitize_label`d.
- **Caller surface (the load-bearing outer contract):** the tool returns MCP
  **text content**, not a raw object. Specify the rendered string: one line per
  result in descending score, e.g.
  `"<score:.3f>  [<space>]  <id>  <sanitized label>  (<file_type>)"`, results
  ordered by score desc and cut to `top_k` after `min_score`/`file_type`
  filtering. The implementer pins this rendered surface, not just the dict.
- **Absent sidecar:** returns the literal instruction text to run
  `graphify embed .` (a normal successful tool result), **not** an exception —
  because `call_tool` wraps exceptions into `"Error executing …"` (`serve.py:1984`)
  which reads to an agent as a malfunction.

---

## 4. Test Targets (red-first)

Each is one behavior. `[new]` = a new-behavior lock (must fail red first before
its slice is implemented). `[guard]` = green on first run; prove it has teeth by
breaking the behavior and confirming the test notices, then keep as a standing
regression guard.

**Text construction (C1):**
- T1 `[new]` — text-family node → `build_node_text` returns exactly
  `"{label}\n{source_file}"`.
- T2 `[new]` — same node, two calls → byte-identical string.
- T3 `[new]` — `image` node → `build_node_text` returns `None` and the node
  appears in **neither** space's ids.

**Enrich → sidecar (C2, C3):**
- T4 `[new]` — after `enrich_embeddings`, `embeddings.npz` exists with keys
  `text_ids`, `text_vecs`, `text_meta`; `text_vecs.dtype == float32`.
- T5 `[new]` — feed the backend stub vectors of norm 5.0; every stored
  `text_vecs` row has `‖v‖ == 1.0` (normalize-on-write).
- T6 `[new]` — `text_meta` parses to JSON carrying `model, backend, dim,
  graphify_version, created_at`; `dim == text_vecs.shape[1]`.

**Search + serve (C4):**
- T7 `[new]` — against a hand-built fixture `.npz`, `semantic_search` returns
  ids in correct descending-cosine order.
- T8 `[new]` — rows with cosine `< min_score` are excluded from the result.
- T9 `[new]` — `file_type` array restricts results to those types.
- T10 `[new]` — every result line carries its `space` (`"text"`).
- T11 `[new]` — a node label containing an injection sentinel is neutralized by
  `sanitize_label` in the rendered output.
- T12 `[new]` — with no sidecar present, the tool returns the
  run-`graphify embed .` instruction and the string does **not** contain
  `"Error executing"`.
- T13 `[new]` — `semantic_search` is present in `list_tools` output and receives
  `project_path` via the existing injection loop (assert the handler is invoked
  with it), **not** that the injection loop itself works — that loop is
  pre-existing, so asserting its generic behavior would be a false lock.
- T14 `[guard]` — construct the server and never call `semantic_search`; assert
  the `.npz` is never opened (lazy load, cached on the graph object). Green on
  first run; break by eager-loading to confirm teeth.

---

## 5. Escalation Map (ordered task breakdown)

Each slice replaces one stub with real behavior, guarded by the §4 targets that
must stay green. Ordering validates the riskiest assumptions first and grows data
real along the pathway; the thread already left search **real** (numpy), so there
is no "make search real" rung.

### Slice R3 — Real text embedding backend (§9)
- **Replaces:** the C2 hardcoded-vector stub.
- **Real behavior:** `EMBEDDING_BACKENDS` dict in `llm.py` (parallel to `BACKENDS`,
  separate); `_call_embeddings` mirroring `_call_openai_compat`'s hardening —
  `input` payload, `data[].embedding` response, `GRAPHIFY_API_TIMEOUT` (600s
  default), **zero SDK retries for ollama** unless `GRAPHIFY_MAX_RETRIES` set,
  batch-capable signature. Backends `ollama` (default) + `openai`; `claude-cli`,
  `bedrock`, `claude`/`anthropic` raise an error **naming the supported set**.
  `OPENAI_BASE_URL` (or embedding-specific equivalent) routes to the **Bifrost
  gateway → local Ollama** — this is the mechanism behind the "cost-free and
  offline" goal, so it is pinned, not prose. Validate returned `dim` against
  `meta.dim`; refuse mismatches.
- **Driving tests (red):** RT1 `EMBEDDING_BACKENDS` exposes `ollama`+`openai`;
  RT2 `claude-cli`/`bedrock`/`claude` raise a message containing the supported
  set; RT3 default ollama issues 0 retries, `GRAPHIFY_MAX_RETRIES=3` overrides;
  RT4 a wrong-dimension vector raises rather than being stored;
  **RT5 (goal-pin)** the default `ollama` backend's resolved base URL is the
  configured local gateway (`OPENAI_BASE_URL`), never a paid cloud endpoint.
  **Keep green:** T4–T6, T7–T12.
- **Done when:** `enrich_embeddings` produces a real `.npz` from a live/local
  text embed, and RT1–RT5 pass with T4–T12 still green. `nomic-embed-text` is
  pulled → low-risk.

### Slice R4 — `.npz` in backup manifest (§3)
- **Replaces:** nothing runtime; a missing registration.
- **Real behavior:** add `embeddings.npz` to `export._BACKUP_ARTIFACTS`
  (`export.py:24`) — it is not regenerable without a re-embed, the criterion that
  list documents.
- **Driving test (red):** RT6 — `_BACKUP_ARTIFACTS` contains `embeddings.npz`.
- **Done when:** RT6 passes; backup path includes the sidecar.

### Slice R5 — Rich deterministic text construction (§2.1/§2.2/§2.3)
- **Replaces:** C1's minimal `"{label}\n{source_file}"`.
- **Real behavior:** code nodes → `{label}` / qualified path / `{rationale?}` /
  space-joined labels of up to 10 highest-degree neighbours; text nodes → prefer
  `rationale`, fall back to `label`+`source_file`, same neighbour augmentation;
  neighbours sorted `(-degree, id)` (total tie-break); **512-token (~2000-char)
  cap truncating neighbours first, never the label.**
- **Driving tests (red):** RT7 output includes label+path+neighbour labels;
  RT8 neighbour order is `(-degree, id)`, tie broken by id, byte-identical across
  two runs; RT9 cap truncates neighbours never the label; RT10 text node prefers
  `rationale`, falls back to `label` when `None`. **Keep green:** T1–T3
  (image still excluded; determinism still holds), and re-embedding still
  produces a valid sidecar.
- **Done when:** RT7–RT10 pass; T1–T3 green; text is a pure function of graph data.

### Slice R6 — Embedding cache + incremental guarantee (§8)
- **Replaces:** the thread's re-embed-everything behavior (no cache).
- **Real behavior:** two namespaces keyed `(backend, model)` under
  `graphify-out/cache/` beside `ast/` and `semantic/`; cache key =
  `sha256(constructed text from §2)` (**not** source-file hash); atomic writes,
  corrupt-entry counting (mirror `cache.py:926`), prune pass mirroring
  `prune_semantic_cache` (`cache.py:1073`).
- **Driving tests (red):** RT11 code and text namespaces independent (switching
  code model leaves text entries hittable); **RT12 (the trap test)** editing
  file A invalidates the entry for a node in file B whose neighbour label
  changed — source-hash keying would wrongly hit; RT13 re-running `embed` on an
  unchanged graph issues **zero** embedding calls; RT14 corrupt `.npz`/entry is a
  miss, is counted, does not raise; RT15 prune drops vectors for nodes no longer
  in the graph. **Keep green:** T4–T6, RT7–RT10 (cache depends on R5 determinism).
- **Done when:** RT11–RT15 pass; the incremental goal (3-file edit ⇒ 3 files of
  calls) is observable.

### Slice R7 — Batching + query-embedding LRU + concurrent query embed (§5/§9)
- **Replaces:** one-input-per-call + single-shot uncached query embedding.
- **Real behavior:** batch `_call_embeddings` inputs per call at a documented
  batch size; in-process LRU cache keyed `(model, text)` for query embeddings;
  issue the (eventually two) query embeddings concurrently.
- **Driving tests (red):** RT16 — 250 nodes at batch size 100 issue exactly 3
  calls; RT17 — the same query twice issues exactly one embedding call per space
  (LRU hit). **Keep green:** T7–T12, RT4.
- **Done when:** RT16–RT17 pass; wall-clock on a 5k-node graph is
  batch-dominated, not per-node.

### Slice R8 — CLI wiring (§6.3)
- **Replaces:** direct `enrich_embeddings(graph)` invocation only.
- **Real behavior:** add `embed` to `cli.dispatch_command`, to the `--help` block,
  and **not** to `_FREE_TEXT_CMDS` (`__main__.py:705`) so the help guard applies;
  `graphify extract . --embed` runs embedding after the graph is written.
- **Driving tests (red):** RT18 `graphify embed .` dispatches; RT19
  `graphify embed --help` shows help rather than running; RT20 `extract --embed`
  embeds after write; RT21 both are a clear no-op message on a zero-node graph.
  **Keep green:** enrich contracts T4–T6.
- **Done when:** RT18–RT21 pass.

### Slice R9 — Neo4j storage tier + prop-filter fix (§7) *(parallel track — see §6)*
- **Replaces:** the silent list-dropping prop filter; no vector index exists.
- **Real behavior:** one shared prop-filter helper applied at **all four sites**
  (node loop, edge loop `graphdb.py:64`, `push_to_falkordb` `:145`/`:162`) that
  allows a `list` whose members are all `int`/`float` (→ Neo4j `LIST<FLOAT>`),
  still drops list-of-dict and other nesting, and checks **`bool` before `int`**.
  Populate `embedding_code`/`embedding_text` node props from the same
  `enrich_embeddings` output (no drift), add a shared `:Embedded` label, and emit
  **two** `CREATE VECTOR INDEX … IF NOT EXISTS` (one per space, dims from
  `meta.dim`, cosine). Skip index creation gracefully on non-Neo4j-5.13+/FalkorDB;
  **a failed index creation must not fail the push.**
- **Driving tests (red):** RT22 a `list[float]` prop reaches the driver; RT23 a
  `list[dict]` is still dropped; RT24 `True` is not coerced to `1`; RT25 all four
  sites use the shared helper; RT26 `CREATE VECTOR INDEX` emitted once per space,
  `IF NOT EXISTS`, and a raised index-creation exception does not fail the push;
  RT27 round-trip against a mock driver — pushed vector equals sidecar vector.
- **Manual acceptance (not unit-testable):** against Aura, a
  `db.index.vector.queryNodes` call returns sensible neighbours — a successful
  push alone does **not** count (§7.1).
- **Done when:** RT22–RT27 pass and the manual Aura queryNodes check returns
  sensible neighbours.

### Slice R10 — Second vector space: code model (§1/§2.1/§5) *(serial — moves contracts)*
- **Replaces:** single-space sidecar/search/result with two spaces.
- **Real behavior:** default code model `nomic-embed-code`; write `code_*` sidecar
  group; code-node text per §2.1; search each space in its own space, take
  per-space top-k, **interleave by raw cosine**, label each result with its
  space, embed the query with **both** models. No cross-space score comparison.
  Gated on the code model being pulled (open question #1); if its dim ≠ 768 the
  per-space `meta.dim` + separate indexes already absorb it.
- **Driving tests (red):** RT28 code-node text includes code-shaped signal;
  RT29 results from both spaces are merged and each still carries its `space`;
  RT30 no assertion of cross-space score comparability; RT31 two embedding calls
  per query (one per space). **Keep green:** T7–T14, RT11 (namespace
  independence), RT16–RT17.
- **Done when:** RT28–RT31 pass with the whole single-space suite still green —
  proving the addition was additive.

### Slice R11 — `query_graph` blend (§6.2) *(serial, gated — moves `_score_query`)*
- **Replaces:** lexical-only seed selection in `_score_query`.
- **Real behavior:** add semantic score as an extra contribution to seed
  selection; **when no sidecar exists the blend weight is zero and scoring is
  bit-identical to today**; a semantic hit never displaces a strong lexical hit
  (exact-label match keeps priority; semantic fills the tail). Blend weight is
  deliberately unspecified (open question #2) — measured against a real graph,
  gated closed by the no-regression test until set.
- **Driving tests (red/guard):** RT32 `[guard]` `query_graph` on a graph with no
  sidecar produces **bit-identical** output to the pre-change implementation
  (capture a golden before the change; this is the reversibility guard — prove
  teeth by perturbing the blend and confirming it fires); RT33 `[new]` an
  exact-label lexical hit still ranks first when a semantic near-miss scores
  higher.
- **Done when:** RT32 stays bit-identical with the feature off and RT33 passes.

---

## 6. Parallelization Matrix

The §4 contract suite is what makes concurrency safe: anything that stays inside
the locked contracts can run independently.

**Safe to parallelize** (contract-protected boundaries):
- **R9 Neo4j tier** — an independent second storage tier + adapter. It consumes
  the same `enrich_embeddings` vector output but touches none of the sidecar or
  `semantic_search` contracts; the prop-filter bug is orthogonal to the thread
  entirely. Only shared dependency: the vector values (needs R1).
- **R4 backup manifest** — a one-line registration behind no runtime contract.
- **R5 rich text construction** — hidden behind the `build_node_text` (C1)
  boundary; search/serve don't care how the string is built as long as vectors
  come out. Runs against the locked sidecar schema.
- **R6 cache** — sits behind the enrich boundary; changes neither the sidecar
  schema nor the search contract. (Depends on R5 for determinism.)
- **R7 batching + LRU** — internal to the backend/query path, contract-preserving.
- **R8 CLI** — a thin adapter over `enrich_embeddings` once its signature is
  locked; the tool surface is unchanged.

**Must stay serial** (these move the boundaries themselves):
- **R1 → R2** — the thread foundation; everything depends on the sidecar +
  tool contracts existing.
- **R3** — replaces the C2 stub on the live thread path; run before anything that
  needs real vectors end-to-end.
- **R10 second vector space** — adds `code_*` sidecar keys and rewrites the search
  merge and result set (interleave, dual query embed). It extends the C3/C4
  contracts, so it is serial by definition; sequence after R1–R3, R5.
- **R11 `query_graph` blend** — alters the `_score_query` scoring path (a
  contract) and is guarded by the bit-identical regression test. Serial and
  gated; sequence after R2.

Anything that would break an existing §4 target is serial by definition.

---

## 7. Invariants (hold across every slice)

- **I1 — Additive & reversible.** A graph with no sidecar behaves exactly as
  today; every new code path is a no-op when the sidecar is absent. *Guarded by:*
  T12 (tool returns instruction, not error) and RT32 (`query_graph` bit-identical
  with the feature off).
- **I2 — Text determinism.** `build_node_text` is a pure function of graph data;
  two runs over an unchanged graph produce byte-identical text, or the cache
  never hits. *Guarded by:* T2, RT8.
- **I3 — Vectors are always L2-normalized on write** (‖v‖ == 1.0), regardless of
  which provider route produced them — so cosine is a plain dot product.
  *Guarded by:* T5.
- **I4 — Spaces are never compared cross-space; every result carries its space.**
  Cosine between a code vector and a text vector is meaningless. *Guarded by:*
  T10, RT29, RT30.
- **I5 — Dimension consistency.** Loaded/returned vectors are validated against
  `meta.dim`; mixing dimensions is refused. *Guarded by:* T6, RT4.
- **I6 — Labels are always sanitized.** Node labels are untrusted LLM output;
  every LLM-derived field in tool output goes through `sanitize_label`.
  *Guarded by:* T11.
- **I7 — Lazy vector load.** A server that never receives a `semantic_search`
  call never opens the `.npz`; the index is cached on the graph object (mirrors
  `_get_trigram_index`). *Guarded by:* T14.
- **I8 — Sidecar is a first-class artifact.** `embeddings.npz` is in
  `export._BACKUP_ARTIFACTS` because it is not regenerable without a re-embed.
  *Guarded by:* RT6.
- **I9 — `bool` before `int`** in every numeric-list membership test in the prop
  filter (`bool` subclasses `int`). *Guarded by:* RT24.
- **I10 — Neo4j and sidecar cannot drift** — both tiers are populated from the
  same `enrich_embeddings` output. *Guarded by:* RT27 (pushed vector == sidecar
  vector).
- **I11 — Offline & cost-free by default.** The default backend resolves to the
  local Bifrost/Ollama gateway (`OPENAI_BASE_URL`), never a paid cloud endpoint;
  excluded backends raise rather than falling through to a 404. *Guarded by:*
  RT5, RT2. (This is the mechanism behind the headline goal — pinned to a test,
  not left in prose.)

---

## 8. Requirement backlog

Flat, ordered, each independently pickup-able by a test-first loop. Read §3
Contracts and §7 Invariants on every one.

- [ ] R1 — Minimal enrich → text-space `.npz` sidecar (C1 minimal, C2 stub, C3, normalize-on-write; T1–T6)  (depends: none; track: serial)
- [ ] R2 — `semantic_search` MCP tool over the sidecar: register + lazy load + brute-force cosine + `min_score`/`file_type`/`top_k` + space label + `sanitize_label` + absent-sidecar instruction + rendered text surface (C4; T7–T14)  (depends: R1; track: serial)
- [ ] R3 — Real text embedding backend: `EMBEDDING_BACKENDS`, `_call_embeddings`, Bifrost/local routing, excluded-backend errors, zero-retry ollama + override, timeout, dim validation (§9; RT1–RT5, I11)  (depends: R1; track: serial)
- [ ] R4 — Add `embeddings.npz` to `export._BACKUP_ARTIFACTS` (§3; RT6, I8)  (depends: R1; track: parallel-with R3,R5,R6)
- [ ] R5 — Rich deterministic text construction: neighbour augmentation `(-degree,id)` top-10, qualified path, rationale-first, 512-token cap (§2; RT7–RT10, I2)  (depends: R1; track: parallel-with R3)
- [ ] R6 — Two-namespace embedding cache keyed on `sha256(constructed text)` + atomic + corrupt-count + prune; incremental guarantee (§8; RT11–RT15)  (depends: R1, R5; track: parallel-with R3)
- [ ] R7 — Batching + query-embedding LRU + concurrent query embed (§5,§9; RT16–RT17)  (depends: R3; track: parallel-with R5,R6)
- [ ] R8 — CLI: `graphify embed .`, `extract --embed`, `dispatch_command`/`--help`/`_FREE_TEXT_CMDS` guard, zero-node no-op (§6.3; RT18–RT21)  (depends: R1, R3; track: parallel-with R5,R6)
- [ ] R9 — Neo4j tier: shared prop-filter helper at 4 sites (bool-before-int, allow `list[int|float]`, drop `list[dict]`) + `embedding_code`/`embedding_text` props + `:Embedded` label + two `CREATE VECTOR INDEX` (graceful skip, non-fatal) + Aura queryNodes acceptance (§7; RT22–RT27, I9,I10)  (depends: R1; track: parallel-with R3,R5,R6,R8)
- [ ] R10 — Second vector space (code model): `nomic-embed-code`, `code_*` sidecar, code text §2.1, dual-space search merge/interleave/space-label, dual query embed (§1,§2.1,§5; RT28–RT31, I4)  (depends: R1, R2, R3, R5; track: serial)
- [ ] R11 — `query_graph` blend: semantic into `_score_query`, gated (weight 0 → bit-identical with feature off), lexical priority (§6.2; RT32–RT33, I1)  (depends: R2; track: serial)

---

## Coverage self-audit

Every spec element has a concrete home (contract C#, invariant I#, slice/backlog
R#, or permanent non-goal):

- **Goals:** retrieve by meaning code+prose → R1/R2 (text) + R10 (code);
  cost-free & offline via local Ollama/Bifrost → **R3 + I11 (RT5)**, the
  means-to-goal pinned, not prose; fully incremental → R6 (RT12/RT13); additive &
  reversible → I1 (T12, RT32); Neo4j carries vectors + queryable index → R9 +
  manual Aura acceptance.
- **Non-goals:** replace lexical / LLM-rerank / HyDE / ANN index / edge-hyperedge
  embedding / node chunking / vector-vs-graph staleness → §2 permanent non-goals.
- **Mechanisms/config/defaults:** two models (`nomic-embed-code`/`-text`) → R1/R3/R10;
  `EMBEDDING_BACKENDS`, ollama default, openai, excluded set → R3; `GRAPHIFY_API_TIMEOUT`
  600s, `GRAPHIFY_MAX_RETRIES`, zero-retry ollama → R3; `OPENAI_BASE_URL`→Bifrost → I11;
  float32 / L2-norm / dim / meta fields / `.npz` path → C3, I3, I5; `_BACKUP_ARTIFACTS`
  → R4/I8; cache dir + `(backend,model)` namespaces + sha256(text) + corrupt-count +
  atomic + prune → R6; 512-token cap + neighbour `(-degree,id)` top-10 + path +
  rationale-first → R5/I2; `semantic_search` schema (top_k/min_score/file_type) +
  `_tools`/`_handlers`/`project_path` injection → C4/R2; `sanitize_label` → I6;
  absent-sidecar instruction → C4/T12; `query_graph` gated blend → R11/I1; CLI
  `dispatch_command`/`--help`/`_FREE_TEXT_CMDS` → R8; prop filter 4-site shared
  helper + bool-before-int + `list[int|float]` allow / `list[dict]` drop → R9/I9;
  `CREATE VECTOR INDEX` ×2 + `:Embedded` + `IF NOT EXISTS` + graceful skip +
  non-fatal failure → R9; lazy load cached on graph → I7; two concurrent query
  embeds + LRU (model,text) → R7; merge interleave / space label / no cross-space
  → I4/R10; batch 250@100=3 → R7 (RT16); dim validation vs meta → I5/RT4.
- **Open questions:** #1 code model not pulled → R10 gated; #2 blend weight
  unspecified → R11 gated by RT32; #3 staleness → permanent non-goal.

No element survives only in prose; the offline/cost-free mechanism is pinned by
RT5/RT2, not left implicit.

---

**Status: ready to hand to a test-first (red/green/refactor) implementer.** Point
the implementer at this file as the authoritative spec. Start at R1; R2 and R3
follow serially; R4/R5/R6/R7/R8/R9 can proceed concurrently against the locked
contracts; R10 and R11 are serial because they move contracts (and R11 stays
gated by its bit-identical regression guard).
