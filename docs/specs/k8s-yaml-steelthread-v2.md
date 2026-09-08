# Steel Thread Plan: YAML/K8s manifest extractor (Graphify fork)

Supersedes: the earlier prose spec ("graphify-yaml-k8s-spec.md") from initial
discussion. Same walking-skeleton scope (plain K8s manifests only — Kustomize,
Argo, CI YAML, Helm all deferred), rewritten in the contracts/requirements
convention used by the embedding-enrichment steel thread so both efforts share
one process. This file is the authoritative spec. §3 Contracts and §6
Invariants are cross-cutting — read them on every requirement's plan step.

---

## 1. Steel Thread Blueprint

The one capability worth shipping first: **list every K8s resource across a
repo (or repos) as graph nodes, with cross-file references (ConfigMap/Secret/
ServiceAccount) correctly resolved and confidence-labeled.** Everything else
(Kustomize, Argo, CI YAML, Helm rendering) is additive depth on the same shape.

K8s manifests are the skeleton, not Kustomize or Argo, because they force the
one hard architectural problem every later dialect will also need: **cross-file
reference resolution**, matching the pattern Graphify already uses for JS/TS
import-gated call resolution (`resolver_registry.py`). Solve it once here.

**What's changed since the original draft:** the multi-repo namespacing problem
this spec originally had to design for is no longer this extractor's concern.
Graphify's `global_add` already prefixes every node id with `repo_tag::`
regardless of which extractor produced it — so this extractor's only
obligation is a stable id *within one extraction run* (see C2). One whole
section of the original draft is now unnecessary.

---

## 2. Deferred (explicitly out of scope for the thread)

- Helm template rendering (unrendered `{{ }}` masks structural parsing).
- Kustomize overlay resolution (`patches`, `components`, remote bases).
- Argo `{{=expr}}` expression parsing.
- CI YAML (GitHub Actions / GitLab CI) job graphs.
- JSON-Schema validation of manifest fields (structural extraction only, not
  semantic correctness).
- Feeding K8sResource nodes into the embedding pipeline (C1/C2 of the
  embeddings plan) — flagged as a cross-project follow-up in §7, not built here.

Each becomes its own future slice, reusing this thread's dialect-detection
pattern and resolver registration — new dialect branches, not new plumbing.

---

## 3. Contracts

### C1 — `detect_dialect(path, raw_text) -> Dialect | None` `[planned]`

Content-based, not extension-based. A file classifies as `K8S_MANIFEST` only
when **every** document in it has both `apiVersion` (str) and `kind` (str) at
the top level, and none matches a more specific signature (Argo's
`apiVersion` prefix `argoproj.io/`, reserved for a future dialect branch).
Anything else returns `None` and falls through untouched.

### C2 — Node id scheme `k8s://{namespace}/{kind}/{name}` `[planned]`

- `{namespace}` is `_cluster` for cluster-scoped kinds or when
  `metadata.namespace` is absent — absence is meaningful (K8s's implicit
  `default` at apply time is never assumed here).
- All components pass through the same NFKC+casefold normalization as
  `ids.py` uses elsewhere.
- **This extractor never encodes repo identity in an id.** Multi-repo
  disambiguation is handled uniformly by `global_add`'s `repo_tag::` prefixing
  after the fact, the same as every other extractor's output — no
  special-casing here, and none should be added later. (Guarded by I3.)

### C3 — Node shape `[planned]`

```json
{
  "id": "k8s://payments/Deployment/api-server",
  "label": "Deployment/api-server",
  "source_file": "manifests/payments/deployment.yaml",
  "source_location": "doc0:L3",
  "attributes": {
    "kind": "Deployment", "namespace": "payments",
    "containers": ["api-server", "sidecar-proxy"],
    "service_account": "payments-sa"
  }
}
```
Containers are attributes, not separate nodes (§5 rationale unchanged from the
original draft). `attributes` tolerance by `validate.py` is an open question
(§7) — confirm before relying on it; fall back to encoding the same info as
edges/separate nodes if rejected.

Any `kind` outside the explicit list still emits a generic `K8sResource` node
(label `"{kind}/{name}"`) rather than being silently dropped (I5).

### C4 — Edge confidence contract `[planned]`

| Relation | Confidence rule |
|---|---|
| `references` (ConfigMap/Secret via envFrom, env[].valueFrom, volumes) | `EXTRACTED` if resolved within the same namespace in the scanned corpus; `AMBIGUOUS` if unresolved (target may exist only at apply time) |
| `uses_service_account` | `EXTRACTED` if resolved; `AMBIGUOUS` if not (K8s silently defaults to `default` SA, never assumed here) |
| `selects` (Service â†’ workload via label match) | Always `INFERRED` — a deduction from label matching, never a stated reference, even when it resolves cleanly |

An `AMBIGUOUS` edge still gets emitted, pointing at a synthesized placeholder
id (`k8s://{ns}/{kind}/{name}#unresolved`) — never silently dropped.

### C5 — Two-pass extraction, resolver-registered `[planned]`

Pass 1 (per file): emit nodes (C3) + *candidate* edges with bare
`target_name`/`target_kind`/`namespace`, kept out of the dict `validate.py`
sees. Pass 2 (whole corpus, registered via `resolver_registry.py`'s
`LanguageResolver` interface — same registration point JS/TS import
resolution uses): build a `(namespace, kind, name) -> node_id` index, resolve
every candidate against it per C4.

---

## 4. Requirements

- [ ] **R1** — Dialect detection (C1) + per-document node emission (C2/C3) for
  plain K8s manifests, generic-fallback for unknown kinds, wired into
  `extract.py`/`detect.py` dispatch. No edges yet — this is "can list every
  resource," the minimal-structure slice, same shape as the embeddings plan's
  R1 (sidecar with no consumer yet).
  `depends: none; track: serial`

- [ ] **R2** — Two-pass resolver (C5): candidate-reference collection +
  `LanguageResolver` registration, producing `EXTRACTED`/`AMBIGUOUS`
  `references`/`uses_service_account` edges (C4). This is the capability the
  whole skeleton exists to prove — the equivalent of the embeddings plan's R2
  (the tool that makes R1's structure useful).
  `depends: R1; track: serial`

- [ ] **R3** — `selects` edges: Service â†’ workload via label-selector match,
  always `INFERRED` (C4). Separate from R2 since it's a distinct heuristic
  (label matching, not reference resolution).
  `depends: R1; track: parallel-with R4`

- [ ] **R4** — Multi-repo composition fixture: two small repos, each with a
  K8s manifest, run through `graphify global add` independently, verify C2's
  claim holds with zero extractor-side changes — ids compose correctly under
  `repo_tag::` prefixing with no special-casing needed. Cheap given the
  contract, but worth a dedicated test given how much of this design leans on
  it working exactly as `global_add` already behaves for other node types.
  `depends: R1; track: parallel-with R3`

- [ ] **R5** — Neo4j push verification for K8s nodes specifically: confirm
  `push_to_neo4j`'s coarse `file_type`-based labeling handles a new
  `k8s`/`iac` `file_type` sensibly, and confirm the Cypher-injection fix
  (issue #84 upstream) actually covers K8s resource names/labels/annotations,
  which carry more punctuation risk (embedded quotes in annotation values,
  etc.) than typical code identifiers.
  `depends: R1, R2; track: serial`

---

## 5. Permanent non-goals (§2 restated as contract, not just deferral)

Same five as the original draft (no Helm rendering, no Kustomize resolution,
no Argo expr parsing, no CI YAML, no schema validation) — unchanged, still
correct, not repeated here for length.

---

## 6. Invariants

- **I1 — Additive.** With no YAML/K8s file present, this extractor is a
  no-op; every other extractor's behavior is unaffected. *Guarded by: R1
  fixture with a non-YAML-only repo.*
- **I2 — Determinism.** Two extraction runs over an unchanged file produce
  byte-identical node/edge dicts. *Guarded by: R1.*
- **I3 — No repo-identity leakage into ids.** Extractor-produced ids never
  contain a repo tag; `global_add` is the only place that adds one. *Guarded
  by: R4's two-repo fixture.*
- **I4 — Confidence integrity.** Every candidate reference resolves to
  exactly one of `EXTRACTED`/`INFERRED`/`AMBIGUOUS`, never silently dropped.
  *Guarded by: R2's resolver tests.*
- **I5 — Unknown kinds never dropped.** Any `kind` outside the explicit list
  still yields a generic `K8sResource` node. *Guarded by: R1's fixture with a
  CRD-shaped resource.*

---

## 7. Open questions

1. Does `validate.py` tolerate the `attributes` key on node dicts (C3), or
   does it need a schema change? Resolve during R1.
2. Does `report.py`/`export.py` handle an edge whose target id has no
   corresponding node (the `AMBIGUOUS` placeholder, C4) without crashing?
   Check directly during R2, don't assume.
3. Cross-project follow-up, not scoped here: once the embeddings plan's R5
   (rich `build_node_text`) lands, should `K8sResource` become a text-family
   node type for embedding purposes (kind/namespace/spec fields as
   constructed text)? Worth a decision once both threads are further along,
   not before.
