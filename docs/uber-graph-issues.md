# Uber-graph: surfaced issues

Findings from walking the stitched 6-repo graph in Neo4j, plus the push/embed
pipeline that fed it. Ordered by severity (correctness first, then gaps, then
throughput).

## Correctness (wrong data in the graph)

### 1. Phantom cross-repo `REFERENCES` edges (HIGH)

`genai-cloud::Client -[REFERENCES]-> upx::CollectionIterator`,
`upx::sync.Mutex -[REFERENCES]-> neo4j-cloud::captureLogger`, and thousands more
`REFERENCES` bind a stdlib/`.`-method symbol in one repo to an *unrelated,
same-named* node in another repo.

- **Where:** the cross-repo merge path — `prefix_graph_for_global` +
  `global_add`/`merge-graphs` compose does not apply the cross-language / phantom
  guard that `build_from_json` applies to *single-repo* `calls`/`imports`/
  `references` (the "Python `import time` must not bind to `time.ts`" rule, #1749,
  and the `.`-method fallback guard in `build.py`). Once ids are repo-prefixed,
  the resolver has nothing to stop ``, `` etc from matching a foreign node.
- **Effect:** any traversal fans out into noise. The upx↔kg-builder "walk"
  appears connected only through these false edges.
- **Fix direction:** carry the phantom-suppression logic (or a `file_type`/
  language-family / explicit-symbol-source check) into the cross-repo
  `REFERENCES` resolution, and drop `REFERENCES` whose target is a `.`-method or a
  builtin that merely shares a name across repos.

### 2. `ISSUES` / `SERVES` = 0 — FIXED (kustomize-slug issuer crashed the TLS pass)

**Final diagnosis + fix (commit `a13aea2`).** A kustomize-emitted `ClusterIssuer`
carries a file-slug id with no `k8s://` (e.g.
`components_cert_manager_k8s_overlays_...`). The TLS pass's
`node_id[len("k8s://"):].split("/", 2)` unpacked one element into three and
raised; `run_language_resolvers` swallows resolver exceptions (resolver_registry.py
`except Exception: _LOG.warning`), so the whole `issues`/`serves` block died
silently while `defines`/`selects`/`references` (earlier in the same pass) landed.
Fix: skip Issuer/ClusterIssuer/Service nodes whose id has no `k8s://`.

Result: `neo4j-cloud` `graph.json` now carries `issues=80, serves=8` (was None).
`serves` is lower than the theoretical 22 because many `spec.dnsNames` targets are
under `.graphifyignore`-d components (see #2b), but the mechanism is proven and
teeth-tested.

`publishes` — **correctly absent**. Of 23 CI workflows across the 6 repos, 20
publish via a shell var (`docker push "$AURA_LATEST_IMAGE"`); the 3 with a
concrete `-t` use non-registry / var-suffixed names. I5 says vars never become
image nodes, so no `publishes` edge is the right answer for this corpus.

### 2b. neo4j-cloud `.graphifyignore` scoped out most components (RESOLVED by config edit)

neo4j-cloud's `.graphifyignore` was a 231-entry deny-list of `components/*/` and
`libs/*/` (generated, "only kg-builder components indexed"). That hid ~84 of 93
cert-manager `Certificate` resources, all their issuers/services, and the bulk of
the Go services. Un-ignoring `components/`+`libs/` (commenting those entries,
`.graphifyignore.bak` kept) grew neo4j-cloud from 18.9k → 131k nodes and put the
full TLS/CRD/k8s subgraph in scope. The edit is uncommitted in the neo4j-cloud repo
(repo policy call: commit it or restore the `.bak`).

### 3. `endpoint://` nodes / `points_to` bridge — FIXED (git-tracked `.env.*` now graphable)

The upx UI calls its backend via runtime `VITE_*` endpoints
(`VITE_AURA_GENAI_API_BASE_URL`, the kg-builder agent API). That is the
`endpoint://<host>` + `points_to` layer — the only legitimate hop from
`upx UI → kg-builder` for the **extraction-from-a-data-source** path (the
local-file path correctly stays intra-upx). Two routing gaps were fixed:

- `.env*` files were skipped as "potentially sensitive" (and unclassified). Now a
  **git-tracked** `.env.<env>` is a committed placeholder and is graphable (commit
  `6f75281`), while bare `.env`/`.env.local`/`.env.*.local`/`.env.*.bak` stay
  excluded. Result: `POINTS_TO=34`, `endpoint://` nodes present.
- Remaining gap: the **host→ingress** join (`endpoint://host` → a foreign-repo
  `Ingress.spec.rules[].host`) is not implemented, and most SaaS hosts
  (`console.neo4j.io` etc.) have no in-corpus Ingress to match anyway.

## Gaps (absent-but-expected, not corrupting)

### 4. `ROUTES` (Ingress→Service) count is 1, expected more

`neo4j-cloud` has ~9 Ingress resources. Most use AWS ALB (`kubernetes.io/ingress.class:
alb` / `alb.ingress.kubernetes.io/...`), so `routes` only fires on the one
`networking.k8s.io` Ingress that declares `backend.service.name`. This is
*semantically* an ALB-host gap (external ingress), not necessarily a bug, but the
Ingress→Service→Workload→Image vertical slice is near-empty as a result. Revisit
when the ALB-ingress or CloudFormation/GCP ingress is in scope.

### 5. `publishes` (CI → image) is entirely absent (see #2)

Same root cause as #2 — the `.github/workflows/*.yaml` files were not extracted, so
the CI publish leg (the thing that would connect CI jobs to the `image://` nodes)
recorded nothing. Confirmed: per-repo `graph.json` files show `publishes=0` even in
`neo4j-cloud` where the workflows exist.

## Throughput / infra

### 6. `push_to_neo4j` was quadratic without an `id` index — FIXED

`MATCH (a {id:$src}), (b {id:$tgt})` per edge, with no index, was a full node scan
per edge (~hours for 265k edges). Fixed in commit `2a9ca71`: a `:GraphifyNode(id)`
range index is created before the edge loop, so edge lookup is O(1). Still
per-edge MERGE (no UNWIND batching), so the push is minutes not seconds; a batched
`UNWIND` transport remains a possible follow-up.

### 7. ollama 7B code-embed wedges under load — documented, not fixed

`nomic-embed-code` (7B) spikes the M-series GPU to ~94% and deadlocked the local
server twice. Switched the pipeline to all-OpenAI (uniform 1536-dim). If local
code embeddings are a requirement, use `nomic-embed-text` or run one repo at a
time and expect restarts. See `docs/uber-graph-setup.md` §3 Variant B.

## Meta

### 8. FYI-only observability gaps (not blocking)

- `graphify global list` / manifest has no embedding-dimension or edge-count
  column — made the "did the four edge families land?" question require manual
  `SHOW`/count Cypher.
- `export neo4j` reads `embeddings.npz` but `global re-embed` writes
  `embeddings-global.npz` — needs the symlink (documented in §7 of the runbook).
