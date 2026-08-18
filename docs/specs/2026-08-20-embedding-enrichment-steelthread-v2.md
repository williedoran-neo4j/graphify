# Steel Thread Plan v2: embedding enrichment + `semantic_search`

Source spec: `/Users/williedoran/Dev/graphify-fork/docs/specs/2026-08-14-embedding-enrichment.md`
Supersedes: `/Users/williedoran/Dev/graphify-fork/docs/specs/2026-08-17-embedding-enrichment-steelthread.md`
(v1). v1's R1–R2 are **DONE**, committed, and locked by tests; this plan keeps
those identifiers and renumbers the remaining scope into R3–R9 (one requirement
per slice). This file is the authoritative spec for the work. §3 Contracts and
§7 Invariants are cross-cutting — read them on **every** requirement.

---

## 1. Steel Thread Blueprint

The one transaction a user (agent) would pay for: **ask a graph a meaning-based
question and get back the right nodes.** The spec's own build note confirms
"Shipping §6.1 alone is a valid first release" — the `semantic_search` MCP tool
is the shippable core; everything else is depth layered onto that shape.

The thread **already exists and is locked** (R1–R2, done):

1. **Enrich.** `enrich_embeddings(graph)` builds a deterministic text string for
   each text-family node (`document`/`paper`/`rationale`/`concept`; `image`
   excluded), embeds it via the `_call_embeddings` stub (deterministic
   `[[i+1]*8]`, no HTTP), **L2-normalizes on write**, and writes a sibling
   `graphify-out/embeddings.npz` holding `text_ids` / `text_vecs` (float32,
   unit-norm) / `text_meta` (`{model, backend, dim, graphify_version,
   created_at}`).
2. **Serve.** `semantic_search` is registered (verbatim §6.1 schema) and picks up
   the existing `project_path` injection. A call **lazily** loads the `.npz`
   (cached on the graph object keyed by `(st_mtime_ns, st_size)`, never touched
   before the first call), scores `text_vecs @ q` (stored rows are unit-norm, so
   cosine is a plain dot product), ranks score-descending, applies `min_score`
   (strict `<` drop) and `file_type` filter, routes every id/label/file_type
   through `sanitize_label`, tags each result `[text]`, and renders the pinned
   MCP text surface. An absent sidecar returns the run-`graphify embed .`
   instruction — never an `"Error executing"` alias.

It is deliberately **one vector space (text) and one backend (stub)**. The
two-model split is "the single most structurally important requirement," so the
sidecar layout, the search core, and the result schema are **space-keyed from
day one** (`text_*` entries, per-line `[space]` tag) — adding `code_*` in R7 is
purely additive, never a rename. `nomic-embed-text` is pulled locally, so the
first escalation rung (real backend) is low-risk; `nomic-embed-code` is not
pulled, gating R7 (open question #1).

**Zero further thread work is needed.** There is no remaining "make the happy
path real" step; the remaining work is depth layered onto the locked shape, one
observable capability per slice.

> **Naming note.** DONE work tracks as "R1" (enrich → sidecar) and "R2"
> (semantic_search tool) in commits and tests, and those identifiers are kept
> here. The NOT-done surface is renumbered R3–R9, one requirement per slice.
> v1's R4 (backup manifest) and v1's R7 (batching/LRU) fold into the new R3;
> v1's R5 (rich text) → R5; v1's R6 (cache) → R4; v1's R9 (Neo4j) → R6; v1's
> R10 (code space) → R7; v1's R11 (blend) → R8; v1's R8 (CLI) → R9, now
> scheduled (see §2). New work only ever *adds to* or *extends* locked contracts.

---

## 2. Deferred (out of scope for the thread)

Every original-spec element lands in a contract, slice, invariant, or permanent
non-goal below — the §"Coverage self-audit" proves it. Nothing is dropped.

**Deferred to a specific slice** (all NOT-done; each becomes one capability):

- **R3 — Real text embedding backend** — `EMBEDDING_BACKENDS`, real
  `_call_embeddings`, Bifrost/local routing (the mechanism behind "cost-free and
  offline"), excluded-backend errors, timeout/retry policy, dim validation,
  batching, the query-embedding LRU, load-time meta validation, and `.npz` in
  `export._BACKUP_ARTIFACTS` (§9, §5 tail, §3).
- **R4 — Embedding cache + incremental guarantee** — two namespaces keyed
  `(backend, model)` under `graphify-out/cache/` (beside `ast/`/`semantic/`),
  cache key = `sha256(constructed text)` (**not** source-file hash), atomic
  writes, corrupt-entry counting, prune pass mirroring `prune_semantic_cache`;
  the headline "3-file edit ⇒ 3 files of calls" invariant (§8).
- **R5 — Rich node text construction** — neighbour labels top-10 `(-degree,
  id)`, qualified path, rationale-first, 512-token (~2000-char) cap (§2.1/§2.2/
  §2.3) + the constructed-text cache-key trap test.
- **R6 — Neo4j storage tier + prop-filter fix** — shared helper at all four
  filter sites (allow `list[int|float]`, drop `list[dict]`, `bool` before
  `int`), `embedding_code`/`embedding_text` properties, shared `:Embedded`
  label, two `CREATE VECTOR INDEX … IF NOT EXISTS` (graceful skip, non-fatal),
  Aura queryNodes manual acceptance (§7). Parallel track.
- **R7 — Second vector space (code model)** — `nomic-embed-code`, `code_*`
  sidecar group, code-node text (§2.1), dual-space search + per-space top-k
  merged/interleaved by raw cosine with a `space` label per result, dual
  concurrent query embeds (§1, §2.1, §5). Serial; gated on the model being
  pulled (open question #1).
- **R8 — `query_graph` blend** — semantic contribution into `_score_query` seed
  selection, gated (weight zero → bit-identical scoring when no sidecar), lexical
  priority preserved (§6.2). Serial; gated by open question #2's unset blend
  weight.
- **R9 — CLI entry points** — `graphify embed .` (standalone re-embed) and
  `graphify extract . --embed` (embed after the graph is written), hand-rolled
  `dispatch_command` wire + `--help` line + `_FREE_TEXT_CMDS` guard + zero-node
  no-op (§6.3; C11). Serial, last — a CLI around a stub is false.

**CLI (§6.3) — now scheduled as R9.** `graphify embed .` and
`graphify extract . --embed` are **restored as a real requirement** by the
user's decision — the previously-recorded re-entry trigger (a user-facing
requirement that `graphify embed .` be runnable) has fired, and open question #4
("CLI disposal") is resolved. R9 stays serial and last because it exposes the
R3 real backend to a user; it is a thin `dispatch_command` branch + `--help`
addition + a zero-node no-op guard, not a new capability shape of its own.

**Permanent non-goals** (never built, §"Non-goals" + open question #3):

1. Replacing lexical search — embeddings *blend into* it (§6.2).
2. LLM re-ranking / HyDE / query expansion.
3. An ANN index (HNSW/IVF) inside graphify — brute force is correct at
   graphify's scale (§4).
4. Embedding hyperedges or edges. Nodes only.
5. Chunking a node's body across multiple vectors. One vector per node.
6. Vector-vs-graph staleness reconciliation (open question #3) — out of scope,
   consistent with the existing deferral of graph-staleness checking.

**(§6.3)** The CLI (`graphify embed .` / `extract --embed`) is **not** a
non-goal. It is scheduled as **R9** (see §5/§8) — scope follows the original
spec §6.3 + §10's `tests/test_embed_cli.py`, and the previous "recorded
re-entry trigger" rationale is superseded by the user's decision to make the
CLI real.

---

## 3. Contracts

Contracts the DONE surface locks are restated briefly, marked `[done]`, and
*kept*: every future slice must keep their tests green. NOT-done contracts are
marked `[planned]`.

### C1 — `build_node_text(node) -> str | None` `[done]`

- Text-family node (`document`/`paper`/`rationale`/`concept`) →
  `f"{label}\n{source_file}"`. Deterministic: two calls, byte-identical (I2).
- `image` (and every non-text-family) node → `None`, excluded from every space.
- R5 *extends* this signature with §2's rich construction. This is a
  **contract alteration to a done contract**: R5's rich form replaces the exact
  `"{label}\n{source_file}"` literal, so the done test T1 (which pins that exact
  string) is *rewritten* in R5 — its label-line-first and determinism halves
  survive as guards, its exact-literal half is superseded by C2 below. Flagged
  because it alters a done contract — see R5.

### C2 — Embedded node text (R5) `[planned]`

R5's richer `build_node_text` output, pinned by structure and content, not by an
exact literal (callers consume vectors, not the string):

- **Structure:** (1) label line, (2) a path-qualified source line, (3) a
  rationale line *when present*, (4) up to 10 space-joined neighbour labels.
- **Ordering:** the label line is the *first* line and survives the cap verbatim;
  neighbours appear in `(-degree, id)` order (total tie-break); the cap truncates
  the neighbour tail, never the label.
- **Content (code, §2.1):** includes the neighbour labels (available graph
  structure, no file reads). **Content (text, §2.2):** `rationale` is the
  primary source; falls back to `label` + `source_file` when absent; same
  neighbour augmentation and cap.
- Determinism (I2) is a hard property — the R4 cache never hits without it.

### C3 — Sidecar `graphify-out/embeddings.npz` `[done]` (R7 extends)

`numpy.savez`, sibling to `graph.json`, space-keyed:

```
text_ids  : (N_t,)      unicode array of node ids
text_vecs : (N_t, D)    float32, every row L2-normalized (||v|| == 1.0)
text_meta : json string {model, backend, dim, graphify_version, created_at}
# R7 adds: code_ids, code_vecs, code_meta   (same shape, reserved in the thread)
```

- `D` recorded in `meta.dim`; loaders validate incoming vectors against it and
  refuse to mix dimensions (I5, RT7).
- Stored rows are always unit-norm (I3): cosine is a plain dot product.
- R7 writes the `code_*` group exactly this way, from §2.1 code text.

### C4 — MCP tool surface `semantic_search` `[done]`

Schema verbatim from §6.1, registered in `list_tools` `_tools` and `_handlers`
with the pre-existing `project_path` injection populated (the injection loop
adds the `project_path` input property; `required` stays `["query"]`).

- **Internal result row:** `{id, label, score, space: "text", file_type}` —
  every LLM-derived field already `sanitize_label`d.
- **Rendered caller surface** (the outer contract tests pin): one line per
  result in descending score,
  `"{score:.3f}  [{space}]  {nid}  {label}  ({file_type})"`, e.g.
  `2.000  [text]  n-b-02  Beta embed  (concept)`. Tests pin values and structure
  (fields, order, the `[space]` literal); cosmetic spacing is not itself the
  contract.
- **Absent sidecar:** a normal successful tool result with the literal
  instruction text "No embeddings found for this graph. Run `graphify embed .`
  first." — never an exception (`call_tool` wraps exceptions into
  `"Error executing …"`).
- **Stale row:** a row whose id is absent from the graph renders id-as-label
  with an empty `()` suffix; never raises (I12).
- **Calls to the same graph root:** after one sidecar load, repeat calls render
  from the in-memory copy; no additional disk open (I7).

### C5 — Embedding backend seam `_call_embeddings` `[planned in R3]`

- Signature `(backend, model, inputs: list[str]) -> list[list[float]]`: exactly
  one vector per input, in input order.
- Backends: `ollama` (default) + `openai`; `claude-cli`/`bedrock`/`claude`
  raise an error naming the supported set (never a fall-through 404).
- `ollama` issues zero SDK retries unless `GRAPHIFY_MAX_RETRIES` is set
  explicitly; `GRAPHIFY_API_TIMEOUT` (default 600s) is honored.
- Batch inputs per call at a documented batch size (I11).
- **Dim guard:** a returned vector of the wrong dimension raises rather than
  being stored (I5).
- Defaults: code → `nomic-embed-code`, text → `nomic-embed-text`, both via the
  `openai`-compatible endpoint (`input` payload, `data[].embedding` response).
- Routing: the default `ollama` backend's base URL is the configured local
  gateway (the embedding-specific `OPENAI_BASE_URL`), never a paid cloud
  endpoint — this is the mechanism behind "cost-free and offline," pinned by RT4.
- Callers must never trust provider normalization — the sidecar writer
  re-normalizes (I3).

### C6 — Embedding write-cache `[planned in R4]`

- Two namespaces keyed `(backend, model)` under `graphify-out/cache/`, beside
  the existing `ast/` and `semantic/` kinds (`cache.cache_dir`). Switching the
  code model must not invalidate text-model entries (RT9).
- A cache entry is keyed by `sha256(constructed text)` from §2 — never a source
  file hash. An edit to file A changes the constructed text of a node in file B
  whose neighbour label changed, so that node's entry misses and is rebuilt
  (RT17 — the trap test).
- Re-running `enrich_embeddings` over an unchanged graph issues **zero**
  embedding calls (RT10) — the incremental goal, falling out of the key.
- Follow the existing cache disciplines rather than inventing new ones:
  corrupt-entry counting (a miss, counted, does not raise — RT11) and a prune
  pass mirroring `prune_semantic_cache` so vectors for nodes no longer in the
  graph do not accumulate forever (RT12). Atomic writes.
- Cache entries are write-path only; sidecar reads (C3/C4's memoization) are
  untouched by it.

### C7 — `query_graph` blend `[planned in R8]`

- When no sidecar exists, `_score_query`'s output is **bit-identical** to
  today's (no semantic contribution, blend weight zero). Observable escape
  hatch: `query_graph` at a fixed input renders a single text; every
  before/after comparison must be byte-equal (RT27).
- A semantic hit never displaces a strong lexical hit (I1): an exact-label match
  keeps rank 1 even when a semantic near-miss scores higher (RT28).
- Blend weight deliberately unspecified (open question #2) — gated closed by
  RT27 until measured.

### C8 — Neo4j tier `[planned in R6]`

- A shared prop-filter helper used at **all four** sites (`push_to_neo4j`
  node+edge, `push_to_falkordb` node+edge): a `list` whose members are all
  `int`/`float` survives to the driver (Neo4j stores `LIST<FLOAT>`); `list[dict]`
  and other nesting still dropped; `bool` checked before `int` (I9).
- Vector node properties `embedding_code` / `embedding_text`, populated from the
  same `enrich_embeddings` output the sidecar gets, so the two tiers cannot
  drift (I10).
- A shared `:Embedded` label on every vector-carrying node; **two**
  `CREATE VECTOR INDEX … IF NOT EXISTS` statements, one per space, dimensions
  from `meta.dim`, cosine. Creation is idempotent, skipped gracefully on
  non-Neo4j-5.13+/FalkorDB, and a failed creation does **not** fail the push
  (RT22).
- **Aura acceptance is explicitly not "the push succeeded":** a manual
  `db.index.vector.queryNodes` call returns sensible neighbours (I13).

### C9 — Search core (graph-agnostic seam) `[done]` (R3 adds load-time validation)

- Returns `None` when the sidecar is absent; a list of `{id, score, space}`
  rows otherwise; score-descending; `min_score` exact-value inclusive; cut to
  `top_k` after ranking.
- R3 adds: on load, validate the sidecar's `meta.dim` against
  `text_vecs.shape[1]` and refuse mismatches (I5, RT7).

### C10 — Result schema, two-space `[planned in R7]`

- Every result carries its own `space` (`code` or `text`); no cross-space score
  comparison is ever asserted. After R7, a fixture whose two spaces both have
  rows above `min_score` and inside `top_k` renders both `[code]` and `[text]`
  lines (RT25), each space's rows score-descending within their space.

### C11 — `graphify embed` / `extract --embed` CLI `[planned in R9]`

Per §6.3, hand-rolled (no argparse subparsers): `__main__.py` reads
`sys.argv[1]` and calls `dispatch_command` (`__main__.py:705`), help text as
literal `print` calls.

- `graphify embed .` — standalone re-embed of an existing graph.
- `graphify extract . --embed` — embed as part of extraction, **after** the
  graph is written.
- `embed` is added to `dispatch_command` **and** to the `--help` block; it is
  **not** added to `_FREE_TEXT_CMDS` (`__main__.py:704`), so the universal help
  guard applies — `graphify embed --help` prints help and stops rather than
  running.
- A **zero-node graph → a clear no-op message**, not an error (§10
  `tests/test_embed_cli.py`).

---

## 4. Test Targets (red-first)

Each is one behavior. `[new]` = new-behavior lock (fails red first); `[guard]` =
green on first run, prove teeth by breaking the behavior (mutator), then keep as
a standing regression guard. **[done]** = already implemented and green today
(v1 R1–R2); never re-schedule.

### The DONE thread — T1–T14 `[done]` (recorded; standing guards for R3–R9)

These live in `tests/test_embed_text.py`, `tests/test_embed.py`,
`tests/test_embed_search.py`, `tests/test_embed_serve.py` and must stay green
through every slice:

- T1 [done] build_node_text text-family node == exactly "{label}\n{source_file}"
  — **note: superseded by C2 when R5 lands (see C1/R5); its determinism and
  label-first halves survive.**
- T2 [done] same node, two calls → byte-identical (I2).
- T3 [done] image node → None; image id absent from text_ids.
- T4 [done] enrich → sibling embeddings.npz; keys exactly {text_ids, text_vecs,
  text_meta}; float32; sorted-id order.
- T5 [done] norm-5 seam feed → every stored row has ||v|| == 1.0 (I3).
- T6 [done] text_meta parses to JSON {model, backend, dim, graphify_version,
  created_at}; dim == text_vecs.shape[1] (I5).
- T7 [done] hand-built orthonormal fixture → rows ranked by cosine
  score-descending (a real permutation of stored order); pinned render
  `"{score:.3f}  [text]  {nid}  {label}  ({file_type})"` in that order.
- T8 [done] min_score drops rows strictly below, keeps rows at the threshold.
- T9 [done] file_type allow-set restricts rows.
- T10 [done] every rendered line carries its "[text]" literal (I4).
- T11 [done] injection sentinel in a node label neutralized by sanitize_label in
  the rendered output (I6).
- T12 [done] absent sidecar → literal run-`graphify embed .` instruction; no
  "Error executing" substring (I1).
- T13 [done] semantic_search in list_tools with exactly the four schema params
  plus project_path injected; required == ["query"].
- T14 [done] building the server never opens the .npz; two semantic_search calls
  on the same server open it exactly once (I7).

### R3 — Real backend, batching, LRU, meta validation, backup manifest (C3/C5/C9)

- RT1 [new] `EMBEDDING_BACKENDS` exposes `ollama` and `openai`.
- RT2 [new] requesting `claude-cli`/`bedrock`/`claude` raises an error naming
  the supported set.
- RT3 [new] `ollama` default issues 0 SDK retries; `GRAPHIFY_MAX_RETRIES=3`
  overrides (observable by counting embedding calls a mocked client makes).
- RT4 [new] **goal-pin (cost-free & offline):** the default `ollama` backend's
  resolved base URL is the configured local gateway (embedding-specific
  `OPENAI_BASE_URL`), never a paid cloud endpoint.
- RT5 [new] batching: 250 inputs at batch size 100 issues exactly 3 calls (I11).
- RT6 [new] query-embedding LRU: the same `(model, text)` query embedded twice
  issues exactly one call per space; a changed text re-calls (§5 tail).
- RT7 [new] load-time dim validation: a sidecar whose `meta.dim` mismatches
  `text_vecs.shape[1]` is refused (raises), a matching sidecar loads (I5).
- RT8 [new] `_BACKUP_ARTIFACTS` contains `embeddings.npz` (I8).

### R4 — Embedding cache + incremental guarantee (C6)

- RT9 [new] namespace independence: switching the active model key leaves
  entries under the previous model key hittable.
- RT10 [new] **zero-call re-run (the incremental goal):** cache-warm
  `enrich_embeddings` over an unchanged graph issues exactly **zero** embedding
  calls.
- RT11 [new] a corrupt cache entry is a miss, is counted, and does not raise.
- RT12 [new] prune drops entries for nodes no longer in the graph (mirroring
  `prune_semantic_cache`).

### R5 — Rich node text + the cache-key trap test (C1/C2/C6)

- RT13 [new] code node: label line first, then path line, then up to 10
  neighbour labels in `(-degree, id)` order; byte-identical across two runs (I2).
- RT14 [new] cap: over-long input truncates neighbours, never the label.
- RT15 [new] text node: rationale present → rationale precedes the neighbour
  block; rationale absent → falls back to label + source_file.
- RT16 [guard] image node still excluded (unchanged-property guard).
- RT17 [new] **the trap test (§8, §10):** editing file A invalidates the cached
  entry for a node in file B whose neighbour label changed — keying on the
  constructed text is correct by construction; a source-file-hash key would
  wrongly hit. (Lands here because the fixture needs neighbour content.)

### R6 — Neo4j tier (C8)

- RT18 [new] a `list[float]` node property survives the shared filter and reaches
  the driver (a bare list member is still dropped; the push does not fail).
- RT19 [new] a `list[dict]` is still dropped.
- RT20 [new] `True` is not coerced to `1` by a numeric-list check (I9).
- RT21 [new] all four filter sites exhibit the shared behavior (node/edge ×
  neo4j/falkordb) — asserted as filter *behavior* at each site, not as an
  implementation detail.
- RT22 [new] `CREATE VECTOR INDEX` emitted once per space, `IF NOT EXISTS`, and
  a raised exception from index creation does **not** fail the push.
- RT23 [new] mock-driver round-trip: the pushed vector equals the sidecar
  vector (I10).

### R7 — Second vector space (C10)

- RT24 [new] code-node text includes the §2.1 code-shaped signal (identifiers/
  signature-adjacent neighbours, not just the bare label).
- RT25 [new] a sidecar carrying both spaces → merged results render both
  `[code]` and `[text]` lines; every line carries its own space; each space's
  rows are score-descending within that space.
- RT26 [new] embedding one query issues two calls — one per space (the §5
  concurrency contract).

### R8 — query_graph blend (C7)

- RT27 [guard] `query_graph` on a graph with **no sidecar** produces
  **bit-identical** output to the pre-change implementation (capture a golden
  before the change; prove teeth by perturbing the blend and confirming it
  fires — I1's no-regression gate).
- RT28 [new] an exact-label lexical hit still ranks first when a semantic
  near-miss scores higher.

### R9 — CLI entry points (C11, §6.3; §10 `tests/test_embed_cli.py`)

The CLI is real behind the **real R3 backend** — not a stub — so these are
`[new]`, unimplemented today. Phrased as observable outcomes, not plumbing:

- RT29 [new] `graphify embed .` dispatches to the embed command (runs the
  re-embed a user asked for).
- RT30 [new] `graphify embed --help` prints help text rather than running —
  the `_FREE_TEXT_CMDS` guard short-circuits before any embedding (observable:
  exit/help printed, no embed side effects).
- RT31 [new] `graphify extract . --embed` runs embedding after the graph is
  written.
- RT32 [new] both commands are **no-ops with a clear message** on a zero-node
  graph — never an error.

Manual acceptance (not unit-testable): against Aura, `db.index.vector.queryNodes`
returns sensible neighbours for a natural-language query (I13) — a successful
push alone does **not** count.

---

## 5. Escalation Map

Each slice is **one behavioral capability**, guarded by the targets above and the
existing done tests. Slices describe what behavior they add in red-first terms;
an implementer expresses each as 1–3 descending test-first steps.

### Slice R3 — Real text embedding backend (C5, §9; folds batches, LRU, meta, manifest)

**Stub replaced:** `_call_embeddings`'s deterministic `[[i+1]*8]` (its first
real HTTP surface, via the same local route the rest of the stack uses).
**Real behavior added:** `EMBEDDING_BACKENDS` in `llm.py` (parallel to
`BACKENDS`, separate); `_call_embeddings` following `_call_openai_compat`'s
hardening (`input` payload, `data[].embedding` response, timeout/retry per §9);
batch sizing (RT5); the query-embedding LRU (RT6); load-time `meta.dim`
validation (RT7); and `embeddings.npz` in `export._BACKUP_ARTIFACTS` (RT8).
`OPENAI_BASE_URL` (embedding-specific equivalent) → **Bifrost gateway → local
Ollama**, the mechanism pinned by RT4. `nomic-embed-text` is pulled locally →
low-risk.
**Driving tests (red):** RT1–RT8. **Keep green:** T4–T6 (sidecar shape and
meta survive the real seam), T7–T12 (search + serve).
**Done when:** `enrich_embeddings` produces a valid unit-norm `.npz` from a
live local text embed; RT1–RT8 pass with the done suite green.
Depends on R1 (done). **Serial** — replaces the live seam everything sits on.

### Slice R4 — Embedding cache + incremental guarantee (C6, §8)

**Stub replaced:** re-embed-everything behavior. **Real behavior added:** two
`(backend, model)` namespaces under `graphify-out/cache/` (via `cache.cache_dir`)
with `sha256(constructed text)` entry keys, atomic writes, corrupt-entry
counting, prune pass mirroring `prune_semantic_cache` (RT9–RT12); the
constructed-text key makes the incremental goal fall out (RT10).
**Driving tests (red):** RT9–RT12. **Keep green:** T4–T6, RT7.
**Done when:** a cache-warm re-embed of an unchanged graph issues zero calls, and
RT9–RT12 pass.
Depends on R3 (a cache around a stub is false; keying also depends on R1's
deterministic text, already done). **Parallel with R5, R6.**

### Slice R5 — Rich deterministic node text construction (C2, §2; supersedes C1's literal)

**Replaces:** `build_node_text`'s minimal `"{label}\n{source_file}"` with the
§2 compose: label line, path line, rationale (when present), up to 10 neighbour
labels `(-degree, id)`, 512-token (~2000-char) cap truncating neighbours first.
**Contract alteration (flagged serial):** this changes a *done* contract — the
done T1 exact-literal pin is superseded. Rewrite T1 in this slice: keep its
determinism and label-first halves (they become T2/RT13-style guards), replace
the exact-string equality with C2's structure assertions. Leave T3 (image) and
T2 (determinism) standing. Extend the text family (rationale-first sourcing,
§2.2) in the same step so RT15 lands. Also lands here: RT17, the §8/§10 trap
test, which needs neighbour-rich text to exist.
**Driving tests (red):** RT13–RT16, RT17. **Keep green:** T2, T3, T4 (sidecar
still writes), T7–T12.
**Done when:** RT13–RT17 pass; text is a pure function of graph data (I2).
Depends on R1 (done). **Parallel with R4, R6.**

### Slice R6 — Neo4j storage tier + prop-filter fix (C8, §7) — parallel track

**Stub replaced:** the silent `isinstance(v, (str, int, float, bool))`
list-dropping filter at four sites (`push_to_neo4j` node+edge,
`push_to_falkordb` node+edge). **Real behavior added:** one shared helper that
allows `list[int|float]` (→ Neo4j `LIST<FLOAT>`), still drops `list[dict]`, and
checks `bool` before `int` (RT18–RT21); `embedding_code`/`embedding_text` props
populated from the same `enrich_embeddings` output (no drift); `:Embedded`
label; two `CREATE VECTOR INDEX … IF NOT EXISTS` (dims from `meta.dim`, cosine,
graceful skip on non-Neo4j-5.13+/FalkorDB, failures non-fatal — RT22); manual
Aura queryNodes acceptance (I13).
**Driving tests (red):** RT18–RT23, then the manual queryNodes check.
**Done when:** RT18–RT23 pass and queryNodes returns sensible neighbours.
Depends on R1 (done) — its only shared dependency is the vector values and the
sidecar schema, both locked. **Parallel with R3, R4, R5.**

### Slice R7 — Second vector space: code model (C3/C10, §1/§2.1/§5) — serial

**Adds:** the `code_*` sidecar group (default model `nomic-embed-code`) written
per §2.1 code text from R5; search each space in its own space, take per-space
top-k, **interleave by raw cosine**, label every result with its own space
(never compare cross-space); embed each query once per space, concurrently.
**Driving tests (red):** RT24–RT26. **Keep green:** T7–T14 (single-space
surface), RT7, RT9 (namespace independence, now two real spaces), RT13.
**Done when:** RT24–RT26 pass with the whole single-space suite green —
proving the addition was additive.
Depends on R3, R5. **Serial** — extends the C3/C4/C10 contracts. Gated on the
code model being pulled (open question #1); a differing dim is absorbed by
per-space `meta.dim` and the separate indexes.

### Slice R8 — query_graph blend (C7, §6.2) — serial, gated

**Adds:** a semantic contribution into `_score_query`'s seed selection, gated to
blend weight zero when no sidecar exists (scoring bit-identical to today) and
lexically-priority-preserving (a strong exact-label hit cannot be displaced).
**Driving tests (red/guard):** RT27 (guard — capture a golden before the change;
prove teeth by perturbing the blend), RT28 (new).
**Done when:** RT27 stays bit-identical with the feature off and RT28 passes.
Depends on R2 (done), R7. **Serial.** Blend weight deliberately unspecified
(open question #2), gated closed by RT27 until measured.

### Slice R9 — CLI entry points (C11, §6.3) — serial, last

**Adds:** the one behavioral capability "the embed path must be runnable by a
user": `graphify embed .` dispatches via `cli.dispatch_command`
(`__main__.py:705`), `embed` is added to `dispatch_command` **and** the `--help`
block **and not** to `_FREE_TEXT_CMDS` (so the universal help guard applies —
RT30), `graphify extract . --embed` calls the embed path after the graph is
written (RT31), and a zero-node graph produces a clear no-op message, never an
error (RT32). Real behavior behind the real R3 backend — a CLI around a stub is
false, because the user-facing contract is "the embed path actually runs end to
end."
**Driving tests (red):** RT29–RT32. **Keep green:** T4–T6 (the sidecar the CLI
produces), T12 (the absent-sidecar instruction now names a runnable command),
the whole done suite.
**Done when:** RT29–RT32 pass from the real CLI and the done suite is green.
Depends on R1 (done), R3. **Serial, last.**

### Slices sensitivity check (why no finer cut)

- **Batching + query-embed LRU + backup manifest** fold into R3: all three are
  the real-call-path's §9/§3 behavior — separate slices would each be a single
  test against the same unbuilt seam.
- **R4 (cache) and R5 (rich text)** stay separate capabilities (construction
  vs. persistence) but run **parallel**; the trap test (RT17) belongs to R5
  because its fixture needs neighbour content, and R4's zero-call guarantee
  (RT10) passes with minimal text.
- **R6 + Aura acceptance** ship together: the queryNodes ability is R6's exit
  criterion, not a separate slice.
- **R7 (code space), R8 (blend), and R9 (CLI)** each stay one capability;
  neither shares slices with the others. Foldables don't exist even at 7
  capabilities: `--help`/dispatch/zero-node behavior are one slice's four
  observable outcomes, not candidates for their own slice, and each R9 target
  below splits no slice-level decision.
- The final count is **7 slices** for the NOT-done surface, one per behavioral
  capability (backend, cache, rich text, Neo4j, code space, blend, CLI). That
  is up from 6 by the user's decision to restore the CLI (§2); still down from
  v1's 11.

---

## 6. Parallelization Matrix

Safe to parallelize (contract-protected — keeps T1–T14 green):

- **R6 (Neo4j tier)** — an independent second storage tier + adapter. It
  consumes the same vector output but touches none of the sidecar or
  `semantic_search` contracts; the prop-filter bug is orthogonal to the thread.
  Only shared dependency: the vector values (R1 done).
- **R4 (embedding cache)** — sits behind the enrich boundary; changes neither
  the sidecar schema nor the search contract (depends on R3 for the real call
  path).
- **R5 (rich text construction)** — hidden behind the `build_node_text` (C1)
  boundary; search/serve don't care how the string is built as long as vectors
  come out. **Caveat, flagged:** R5 *rewrites* the done T1 exact-literal (a
  contract alteration to C1) and owns the RT17 trap test. That makes R5 the one
  parallel slice with a coordination obligation: R4 must not claim the trap
  test, and R4's zero-call guarantee (RT10) is only meaningful against R5's
  deterministic key — an ordering note, not a serialization.

Must stay serial (these move the boundaries themselves):

- **R3** — replaces the live seam; everything that needs real vectors
  end-to-end stands behind it.
- **R7 (second vector space)** — extends the C3/C4/C10 contracts (adds `code_*`
  keys, rewrites the merge and result set). Serial by definition; sequence after
  R3, R5.
- **R8 (query_graph blend)** — alters the `_score_query` scoring path (a
  contract) and is guarded by RT27's bit-identical regression test. Serial and
  gated; sequence after R7.
- **R9 (CLI)** — exposes the real embed path to a user; it depends on R3's real
  `_call_embeddings` and R1's sidecar write, so a wrong sequence (a CLI round a
  stub) would lock a false shape. Serial and last; sequence after R3.

Anything that would break an existing T1–T14/RT target is serial by definition.
The DONE thread needs no parallel track — it is the finished core every track
hangs off.

---

## 7. Invariants

Hold across every slice; the test guarding each is named.

- **I1 — Additive & reversible.** A graph with no sidecar behaves exactly as
  today; every new code path is a no-op when the sidecar is absent. *Guarded by:
  T12 (done), RT27 (R8).*
- **I2 — Text determinism.** `build_node_text` is a pure function of the graph's
  data; two runs over an unchanged graph produce byte-identical text, or the
  cache never hits. *Guarded by: T2 (done), RT13 (R5).*
- **I3 — Vectors are always L2-normalized on write** (||v|| == 1.0), regardless
  of which provider route produced them — cosine is a plain dot product.
  *Guarded by: T5 (done).*
- **I4 — Spaces are never compared cross-space; every result carries its own
  space.** Cosine between a code vector and a text vector is meaningless.
  *Guarded by: T10 (done), RT25 (R7).*
- **I5 — Dimension consistency.** Loaded/returned vectors are validated against
  `meta.dim`; mixing dimensions is refused. *Guarded by: T6 (done), RT7 (R3).*
- **I6 — Labels are always sanitized.** Every LLM-derived field in tool output
  goes through `sanitize_label`. *Guarded by: T11 (done).*
- **I7 — Lazy vector load.** A server that never receives a `semantic_search`
  call never opens the `.npz`; the loaded sidecar is cached on the graph object.
  *Guarded by: T14 (done).*
- **I8 — Sidecar is a first-class artifact.** `embeddings.npz` is in
  `export._BACKUP_ARTIFACTS` because it is not regenerable without a re-embed.
  *Guarded by: RT8 (R3).*
- **I9 — `bool` before `int`** in every numeric-list membership test in the prop
  filter (`bool` subclasses `int`). *Guarded by: RT20 (R6).*
- **I10 — Neo4j and sidecar cannot drift** — both tiers are populated from the
  same `enrich_embeddings` output. *Guarded by: RT23 (R6).*
- **I11 — Batch discipline.** Embedding calls are batched at a documented size;
  wall-clock is dominated by batches, never one HTTP round-trip per node.
  *Guarded by: RT5 (R3).*
- **I12 — Stale sidecar rows never raise.** A row whose id is absent from the
  graph renders id-as-label with an empty `()` suffix. *Guarded by: the stale-row
  test in `tests/test_embed_search.py` (done).*
- **I13 — Neo4j vectors must be *queryable*, not just pushed.** A round-trip
  that does not answer `db.index.vector.queryNodes` is a silent failure.
  *Guarded by: manual Aura acceptance (R6); a successful push alone does not
  count.*

---

## 8. Requirement backlog

Flat, ordered, each independently pickup-able by a test-first loop. Read §3
Contracts and §7 Invariants on every one. K-format: one line per requirement,
marking the DONE ones `[x]`.

- [x] R1 — Minimal enrich → text-space `.npz` sidecar: `build_node_text`
  text-family selector + `_call_embeddings` stub + `enrich_embeddings`
  (normalize-on-write, meta)  (depends: none; track: serial) — DONE
- [x] R2 — `semantic_search` MCP tool over the sidecar: registration + lazy
  graph-object-hosted load + brute-force cosine + min_score/file_type/top_k +
  space label + sanitize_label + absent-sidecar instruction + rendered surface
  (C4)  (depends: R1; track: serial) — DONE
- [ ] R3 — Real text embedding backend + batching + query-embed LRU +
  load-time meta validation + backup manifest: `EMBEDDING_BACKENDS`, real
  `_call_embeddings`, Bifrost/local routing (RT4), excluded-backend errors,
  0-retry ollama + override, timeout, dim guard, batch sizing, LRU,
  `embeddings.npz` in `export._BACKUP_ARTIFACTS` (C5/C9, §9/§5/§3;
  RT1–RT8, I5/I8/I11)  (depends: R1; track: serial)
- [ ] R4 — Embedding cache + incremental guarantee: two `(backend, model)`
  namespaces, `sha256(constructed text)` keys, atomic writes, corrupt-count,
  prune pass (C6, §8; RT9–RT12; the "3-file edit ⇒ 3 files of calls" goal)
  (depends: R3; track: parallel-with R5, R6)
- [ ] R5 — Rich deterministic node text construction: label/path/rationale/10
  neighbours `(-degree, id)`, rationale-first, 512-token cap; rewrites the done
  T1 literal per C2; hosts the cache-key trap test (C1/C2, §2; RT13–RT17, I2)
  (depends: R1; track: parallel-with R4, R6)
- [ ] R6 — Neo4j tier: shared prop-filter at all 4 sites (bool-before-int,
  allow `list[int|float]`, drop `list[dict]`) + `embedding_code`/`embedding_text`
  props + `:Embedded` + two `CREATE VECTOR INDEX` (graceful, non-fatal) + Aura
  queryNodes manual acceptance (C8, §7; RT18–RT23 + manual, I9/I10/I13)
  (depends: R1; track: parallel-with R3, R4, R5)
- [ ] R7 — Second vector space (code model): `nomic-embed-code`, `code_*`
  sidecar, code-node text §2.1, dual-space search merge/interleave/space-label,
  dual concurrent query embed (C3/C10, §1/§2.1/§5; RT24–RT26, I4)
  (depends: R3, R5; track: serial)
- [ ] R8 — `query_graph` blend: semantic into `_score_query`, gated (weight 0 →
  bit-identical), lexical priority (C7, §6.2; RT27–RT28, I1)
  (depends: R2, R7; track: serial)
- [ ] R9 — CLI: `graphify embed .` + `extract --embed` (dispatch_command and
  `--help` block, `_FREE_TEXT_CMDS` guard so `--help` never runs, zero-node
  no-op message: C11, §6.3; RT29–RT32)
  (depends: R1, R3; track: serial)

---

## Coverage self-audit

Every original-spec element has a concrete home — a contract (C#), invariant
(I#), slice/backlog (R#), or permanent non-goal — and every "means to a goal"
mechanism is pinned to a test, not left in prose.

**Goals (§"Goals"):**
- Retrieve by meaning for code and prose → R1/R2 (text, done) + R7 (code).
- Cost-free and offline by default (local Ollama via Bifrost) → **R3 + RT4**
  (the base-URL routing is pinned to a test, not prose).
- Fully incremental (re-embedding a 5k-node graph after a 3-file edit costs 3
  files of calls) → R4 (RT10) + R5 (RT17 — key correctness).
- Additive and reversible (no-embeddings graph behaves exactly as today) → I1
  (T12 done, RT27 in R8).
- Neo4j push carries vectors + creates a queryable vector index (Aura
  first-class) → R6 + manual Aura acceptance (I13).

**Non-goals (§"Non-goals"):** replace lexical → blend, not replacement (R8);
LLM re-ranking / HyDE / query expansion → non-goal #2; ANN index (HNSW/IVF)
inside graphify → non-goal #3; embedding edges/hyperedges → non-goal #4;
chunking a node across multiple vectors → non-goal #5.

**§1 Scope table + two-model requirement:** code → code model; document/paper/
rationale/concept → text model; image → excluded (T3 done). Defaults
`nomic-embed-code`/`nomic-embed-text` → C5. "Vectors from different models are
not comparable" → I4 (T10 done, RT25 in R7), which is what drives "two cache
namespaces" (R4), "two vector indexes" (R6), and "per-space search" (R7). The
space-keyed sidecar layout (C3) is the structural spine that makes the second
model additive.

**§2 Node text construction:** start-line-only `source_location` (no body
extent, no extractor change) → C2 documents the composed form explicitly *not*
using body text; code composition (label/path/rationale/neighbours) → C2/RT13;
neighbour labels load-bearing + 512-cap (truncate neighbours first, never the
label) → C2/RT14; text nodes rationale-first, fall back to label+source_file →
C2/RT15; determinism `(-degree, id)` total tie-break → I2/RT13.

**§3 Storage:** `embeddings.npz` sibling, not `graph.json` (diffable, 512 MiB
cap, incremental rewrite broken by inlining) → C3 done (T4); float32 not
float64 + stored pre-normalized + defensive re-normalize on write → C3/I3
(T5 done); `.npz` in `export._BACKUP_ARTIFACTS` → I8/RT8 (R3); §3.1 both tiers
populated from the same `enrich_embeddings` output → I10/RT23 (R6).

**§4 Search:** numpy brute force, no new dependency, scale argument → T7 done
(vector math is numpy's); lazy load + cached on the graph object mirroring
`_get_trigram_index` → I7/T14 done.

**§5 Query-side routing:** two embedding calls per query (both local, ~15 ms,
issued concurrently) → R7 (RT26); query-embedding LRU keyed `(model, text)` →
R3 (RT6 — folded into the real-call slice, batch-sized); per-space top-k merged
by interleaving raw cosine, each result labeled with its space, no cross-space
score comparability claim → C10/R7 (RT25) + I4.

**§6 Integration surfaces:** §6.1 schema/registration/project_path/sanitize
labeling/absent-sidecar-instruction → C4, T11–T13 done; §6.2 blend + gating +
lexical priority + open question #2 → C7/R8 (RT27/RT28); §6.3 CLI → C11/R9
(RT29–RT32; open question #4 resolved — see §2).

**§7 Neo4j:** prop-filter silent list-drop + "four sites, one shared helper" +
`bool`-before-`int` → C8/R6 (RT18–RT21); `embedding_code`/`embedding_text` +
`:Embedded` + two `CREATE VECTOR INDEX` + `IF NOT EXISTS` + graceful skip +
non-fatal failure → C8/R6 (RT22); Aura queryNodes = not-just-pushed acceptance →
I13 (manual).

**§8 Cache design:** two namespaces keyed `(backend, model)` beside `ast/`/
`semantic/` → R4 (RT9); key = `sha256(constructed text)`, not source hash, with
the file-B-neighbour rationale → R4/R5 (RT17 trap test, RT10 zero-call);
corrupt-entry counting + atomic writes + prune mirroring `prune_semantic_cache`
→ R4 (RT11, RT12, write discipline).

**§9 Backend wiring:** `EMBEDDING_BACKENDS` parallel to `BACKENDS` but separate
→ C5/R3; excluded backends `claude-cli`/`bedrock`/`claude` fail with the
supported set named → RT2; zero SDK retries for ollama unless
`GRAPHIFY_MAX_RETRIES` set + `GRAPHIFY_API_TIMEOUT` (600s) → C5/RT3; batch
sizing → I11/RT5; dim validation against `meta.dim` → I5/RT7; `OPENAI_BASE_URL`
→ Bifrost → RT4.

**§10 Acceptance criteria, test-module by test-module:**
- `test_embed_text.py` — T1–T3, all done (T1's exact literal superseded by R5
  per C1; determinism and image-exclusion halves continue).
- `test_embed_backends.py` (does not yet exist as a module) — §9 bullets →
  R3's RT1–RT5, RT7.
- `test_embed_cache.py` (does not yet exist as a module) — namespace
  independence → RT9; constructed-text invalidation → RT17; zero-call re-run →
  RT10; corrupt-npz/entry is a miss, counted, does not raise → RT11; prune →
  RT12.
- `test_embed_search.py` — cosine ordering ✓ done (T7); min_score ✓ (T8);
  file_type ✓ (T9); space ✓ (T10); query-LRU → R3 RT6 (does not require two
  spaces — one space, two texts, one call each); lazy load ✓ (T14).
- `test_embed_serve.py` — registration ✓ (T13); absent-sidecar ✓ (T12);
  sanitize ✓ (T11); query_graph bit-identical → R8 (RT27); exact-label priority
  → R8 (RT28).
- `test_embed_graphdb.py` — all §7 bullets → R6 (RT18–RT23).
- `test_embed_cli.py` — §6.3 bullets → C11/R9 (RT29–RT32).
- Manual Aura acceptance → I13.

**§11 Build order:** graphdb-first → R6 (parallel); llm → R3; cache → R4; embed
text → R1 done; serve → R2 done + R8; cli → R9 (depends R1+R3). R7 (code space)
depends on R3+R5; R8 (blend) depends on R2+R7. Ordering preserves "tests pass
before the next begins" by making every dependency explicit rather than implicit.

**§12 Open questions:** #1 (code model not pulled; dim variance absorbed by
per-space `meta.dim` + separate indexes) → R7 (gated) + C5; #2 (blend weight
unspecified) → R8 (gated by RT27); #3 (vector/graph staleness out of scope) →
permanent non-goal #6; #4 (CLI disposal — added in v2) → resolved: the CLI is
scheduled as R9 per the user's decision.

**What the previous v1 plan's §2 "Deferred" section had — and where it went:**
real backend → R3; rich text → R5; second space → R7; embedding cache +
incremental → R4; batching + LRU → R3; CLI → R9 (restored as a real requirement);
query_graph blend → R8; Neo4j tier → R6; `.npz` in `_BACKUP_ARTIFACTS` → R3 (RT8); permanent
non-goals → §2 #1–#7. Nothing from v1's deferred list (or the original spec) is
dropped.

---

## Over-granularity check (explicit)

Target total, walking §4: 8 (R3) + 4 (R4) + 5 (R5) + 6 (R6) + 3 (R7) + 2 (R8) =
**28 new-behavior red targets** across **6 slices** (≈4–5 per slice, none over
6), plus the 14 done tests as standing guards and 1 manual acceptance.

- **Per-contract budget:** the largest contract lists are C5 (RT1–RT6 — the
  backend seam) and C8 (RT18–RT23 — the Neo4j tier). Each is genuinely several
  sub-contracts (registry/routing vs. call behavior; filter vs. index vs.
  drift), split inside its slice rather than inflated as one behavior; no
  single contract exceeds 6.
- **Guard batching:** exactly one guard target exists (RT27, the R8
  no-regression golden) and it enters the loop as a teeth-check (mutator
  perturbing the blend), not a serialized implementation phase. R5's RT16 is a
  folding of the done T3's unchanged-property into that slice's single
  keep-green note.
- **Observed at the caller boundary, never the plumbing:** no target asserts
  "the handler is invoked with X", "this file opens N times", or "the cache is
  keyed by …" as a mechanism — every target pins observable outcomes (returned
  renders, call counts as a *measured* side effect of the real path, byte-equal
  texts, hash-stable re-entries). The one deliberate near-mechanism assertion
  (RT21 "all four sites share the behavior") is phrased as filter *behavior at
  each site*, per the original spec's explicit four-site acceptance.
- **Value/struct pinned, not cosmetics:** renders are pinned by field/order/
  literal (`[space]`), never by spacing beyond what's already locked; the text
  cap is pinned by what *survives* (the label), not a character count of the
  whole string.
- **Slices vs. capabilities:** 7 slices = 7 behavioral capabilities (backend,
  cache, rich text, Neo4j, code space, blend, CLI), one per slice — up from
  v1's 11, and 7 instead of 6 because the user restored the CLI (§2) as R9.
  The audit found nothing further to fold: splitting R4 from R5, R6 from R3,
  or R7 from R8 would each yield a slice whose only content is a single test
  name, which the skill forbids.

The plan is deliberately at this granularity; the check changes nothing.

---

**Status: ready to hand to a test-first (red/green/refactor) implementer.** R1
and R2 are already done and locked — an implementer picking this up cold starts
at R3 (serial foundation), then R4/R5/R6 in parallel against the locked
contracts, then R7 and R8 serially. Read §3 Contracts and §7 Invariants on every
requirement; §4's done targets (T1–T14) are the standing regression suite.
