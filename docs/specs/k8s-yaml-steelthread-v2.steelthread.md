# Steel Thread Plan: YAML/K8s manifest extractor (Graphify fork)

Supersedes `k8s-yaml-steelthread-v2.md` as the **authoritative implementer spec**.
Produced by the steel-thread skill on 2026-08-26, grounded against the actual
Graphify source. Where the source spec's assumptions contradicted the code, this
plan records the contradiction and picks one reading — each is called out with a
`[CORRECTION]` marker so nothing looks like a silent edit.

Cross-cutting §3 (Contracts) and §7 (Invariants) must be readable on **every**
requirement. §8 is the one-at-a-time backlog; an implementer sees only the single
requirement it is working.

---

## 0. Grounding summary — what the source spec got wrong

The source spec is structurally sound (same contracts/requirements convention as
the embeddings thread) but five of its load-bearing assumptions do not survive
contact with the code. These are resolved here, not deferred:

1. **`.yaml` never reaches the AST extractor.** `detect.py` classifies `.yaml`/`.yml`
   as `DOCUMENT` (they live in `DOC_EXTENSIONS`, detect.py:44-45), and the CLI
   routes `document` files to the **LLM/semantic** path (cli.py:3388-3398), while
   only `code` files go to the AST extractor (`_ast_extract`, cli.py:3679). The
   natural seam for a K8s extractor is `extract.py:_DISPATCH` (a suffix→extractor
   map, mirroring `.tf`→`extract_terraform` at extract.py:5245), but that seam is
   only ever consulted for `code` files. **R1 must move `.yaml`/`.yml` into the
   code path before any K8s extraction runs.** *Decision confirmed with author:
   all `.yaml`/`.yml` move to the AST path; non-K8s YAML becomes a no-op.* See C0.

2. **`file_type` is required and whitelisted — the C3 node shape violates it.**
   `validate.py:6` *requires* `file_type` on every node and `validate.py:4`
   whitelists it to `{code, document, paper, image, rationale, concept}`.
   Independently, `build.py:856` silently coerces any unknown `file_type` to
   `"concept"`. The C3 example omits `file_type` (→ validation error) and R5's
   `k8s`/`iac` type is in neither whitelist (→ would degrade to `Concept`). **The
   `k8s` node type must be added to both whitelists**; done as part of R1.

3. **The C4 "placeholder id with no node" edge is dropped, not emitted.**
   `validate_extraction` flags an edge whose endpoint matches no node id
   (validate.py:73-85); `build.py:875-877` demotes that to a warning, and
   `build.py:1162-1163` **drops the edge before it reaches the graph/report/Neo4j**.
   So "emit an `AMBIGUOUS` edge pointing at `…#unresolved`" as written produces
   *nothing*. **The placeholder must be a real synthesized node**, which the
   resolver materializes in pass 2. See C4.

4. **The C2/C3 id scheme contradicts `ids.py` normalization.** `ids.normalize_id`
   (ids.py:50-83) casefolds and collapses every non-word char (`/`, `:`, `#`,
   `-`) to `_`; `ids.make_id` (ids.py:86) is the only id constructor any current
   extractor uses (`extractors/base.py:54`). It would turn
   `k8s://payments/Deployment/api-server` into `k8s_payments_deployment_api_server`
   — and casefold `Deployment`→`deployment`. The C3 example preserves `/`, `:`,
   case, and the `-` in `api-server`, which is impossible through `make_id`. **The
   K8s extractor emits raw id strings (bypassing `make_id`); build and `global_add`
   pass them through byte-identically** (verified: neither re-normalizes node ids —
   build only uses `_normalize_id` for *edge-endpoint reconciliation*, build.py:1065/1159,
   and `prefix_graph_for_global` prepends `repo_tag::` verbatim, build.py:1999).

5. **R5 is verification-only, and the injection concern is already moot on the
   live push path.** `exporters/graphdb.py` builds the Neo4j label generically
   from `file_type.capitalize()` (graphdb.py:186-192) — a new type "just works"
   *once whitelisted upstream*. All node/edge **property values are sent as Cypher
   query parameters** (`SET n += $props`, `$id`, `$props`), never interpolated
   (graphdb.py:186-207), so embedded quotes/backslashes/newlines/braces in K8s
   annotation values are safe. The `to_cypher` file-writer path escapes literals and
   allowlists identifiers (export.py:430-472) and never writes arbitrary attribute
   values at all. **R5 = "assert it's parameterized + label `K8s` is produced",
   no new plumbing.**

Two of the source spec's §7 open questions are now answered by grounding:
Q1 (`validate.py` tolerate `attributes`?) → **yes**, unknown node keys are ignored;
the real gate is `file_type`. Q2 (report/export crash on placeholder target?) →
**unreachable** — the edge is dropped upstream, which is why C4 becomes a node.

---

## 1. Steel Thread Blueprint

**List every K8s resource across a repo as graph nodes, with cross-file
references (ConfigMap/Secret/ServiceAccount) resolved and confidence-labeled.**

One flow, thin end to end:

1. A `.yaml`/`.yml` file is classified as **code** (C0), so the CLI hands it to
   the AST extractor.
2. `extract.py:_DISPATCH` routes `.yaml`/`.yml` → `extract_k8s`.
3. `extract_k8s` content-gates each file with `detect_dialect` (C1): a K8s
   manifest parses and emits one node per top-level document (C2/C3, raw id
   scheme, `file_type="k8s"`, containers as attributes); anything else emits
   nothing (the additive I1 no-op).
4. Pass 1 also stashes *candidate* references in a per-file side-channel; a
   registered `LanguageResolver("k8s_yaml", …)` (C5, mirroring the Swift/TS
   member-call resolvers) runs pass 2 inside `extract()` and resolves them to
   `references` / `uses_service_account` edges (EXTRACTED/AMBIGUOUS), synthesizing
   a real placeholder node per unresolved target (C4).
5. `selects` edges (Service→workload, always INFERRED) are a separate heuristic
   on the same skeleton (C4/R3).
6. The resulting graph survives `validate` + `build` (C2/C3 whitelist), and
   `global_add` prefixes ids uniformly (R4) with zero extractor-side changes.

The hard architectural problem this forces — cross-file reference resolution — is
the same one every later dialect (Kustomize, Argo, CI YAML) will need, and it
plugs into the **already-built** `resolver_registry.py` seam JS/TS import
resolution uses. Containers stay attributes, not nodes (rationale unchanged from
the source spec): a container is not a cross-file reference target and would add
an edge class we then never resolve.

## 2. Deferred (explicitly out of scope for the thread)

- Helm template rendering (unrendered `{{ }}`).
- Kustomize overlay resolution (`patches`, `components`, remote bases).
- Argo `{{=expr}}` expression parsing.
- CI YAML (GitHub Actions / GitLab CI) job graphs, docker-compose, Ansible.
- JSON-Schema validation of manifest fields (structural extraction only).
- Feeding `K8sResource` nodes into the embedding pipeline (they must **not** be
  added to `embed._TEXT_FILE_TYPES` — verified that `file_type="k8s"` → no vector,
  which is the desired behavior; see C3 note).
- **[REGRESSION NOTE — accepted]** Under C0, *all* `.yaml`/`.yml` move off the LLM
  document path. Repos that today index CI/docker-compose/config YAML as document
  nodes will **lose those nodes**. This is deliberate (author-confirmed): YAML
  coverage will be re-added dialect-by-dialect once the K8s thread lands. Guarded
  by I6 so it's visible, not accidental.

Each deferred item is a future slice reusing this thread's `detect_dialect` +
`_DISPATCH` + resolver registration — new dialect branches, not new plumbing.

---

## 3. Contracts

### C0 — YAML routing into the AST path `[planned]`

`.yaml` and `.yml` are classified as **code**, not document.

- `detect.py`: remove `.yaml`/`.yml` from `DOC_EXTENSIONS`, add to `CODE_EXTENSIONS`.
- `extract.py:_DISPATCH`: add `".yaml": extract_k8s, ".yml": extract_k8s`.
- Consequence: `code_files` picks up YAML (cli.py:3388) → `_ast_extract`
  (cli.py:3679) → `_DISPATCH["…"]` → `extract_k8s`.
- Precedence unchanged: `is_package_manifest_path` still intercepts `apm.yml` /
  `apm.yaml` before suffix dispatch (extract.py:5402, manifest_ingest.py:28-34) —
  no collision, K8s manifests are not named `apm.yml`. `_get_extractor` lowercases
  a bare suffix (extract.py:5409) so `.yml`/`.yaml` work as-is.

### C1 — `detect_dialect(path, raw_text) -> Dialect | None` `[planned]`

Content-based, not extension-based.

- A file classifies as `K8S_MANIFEST` iff **every** top-level document in
  `raw_text` (multi-doc parse via `yaml.safe_load_all`, split on `---`) has both
  `apiVersion` (str) and `kind` (str) at the top level, **and** no document's
  `apiVersion` starts with `argoproj.io/` (reserved for a future Argo branch).
- Returns `None` for: empty text, unparseable YAML, a top-level document that is
  not a mapping, any doc missing `apiVersion`/`kind`, or any `argoproj.io/*`
  `apiVersion`.
- Deterministic (I2); `path` is used only for diagnostics, classification reads
  `raw_text` alone.
- `Dialect` is a small enum: `K8S_MANIFEST` today; a later slice adds branches
  without touching call sites.

### C2 — Node id scheme (raw emission) `[planned]` `[CORRECTION]`

- `id = f"k8s://{namespace}/{kind}/{name}"` — **verbatim**, emitted as a raw
  string, **not** passed through `ids.make_id`/`normalize_id`.
  - `namespace` = `metadata.namespace` if present, else `"_cluster"`. Absence is
    meaningful; K8s's implicit `default` is never assumed. Cluster-scoped kinds
    are *automatically* `_cluster` because they never carry `metadata.namespace`
    — no kind list is hardcoded.
  - `kind` = the manifest `kind` string, `name` = `metadata.name`, both verbatim
    (case preserved). K8s `kind`/`name` are case-sensitive; casefolding would
    collapse distinct resources (`Pod`/`pod`, `apiServer`/`ApiServer`).
- `[CORRECTION]` The source spec's "same NFKC+casefold normalization as `ids.py`"
  clause is **rejected**: `normalize_id` (ids.py:81) casefolds and collapses `/`,
  `:`, `-` to `_`, which would both contradict the C3 example and merge distinct
  K8s resources. The verbatim scheme is deterministic (I2) and matches how K8s
  itself addresses resources. If the author later wants `ids.py`-normalized ids,
  it is a one-line contract change (route components through `make_id`) — but that
  loses the `k8s://` shape the R4 test depends on, so it is not the default.
- Unresolved-target placeholder id (C4): `f"k8s://{namespace}/{kind}/{name}#unresolved"`.
- **This extractor never encodes repo identity in an id.** Multi-repo
  disambiguation is `global_add`'s `repo_tag::` prefix (global_graph.py:117,
  build.py:1999). No extractor-side special-casing now or later. (I3.)

### C3 — Node shape `[planned]` `[CORRECTION: adds file_type]`

```json
{
  "id": "k8s://payments/Deployment/api-server",
  "label": "Deployment/api-server",
  "file_type": "k8s",
  "source_file": "manifests/payments/deployment.yaml",
  "source_location": "doc0",
  "attributes": {
    "kind": "Deployment",
    "namespace": "payments",
    "containers": ["api-server", "sidecar-proxy"]
  }
}
```

- **`file_type` is `"k8s"`, required** — `[CORRECTION]` the source spec omitted it.
  Both whitelists are extended in R1: `validate.py:4` (`VALID_FILE_TYPES += {"k8s"}`)
  and `build.py:856` (add `"k8s"` to the allowed set), else the node validates
  invalid *and* degrades to `Concept`.
- `label = f"{kind}/{name}"`.
- `source_location = "doc{i}"` — zero-based top-level document index. The
  `:L{line}` line-position suffix is **optional decoration**, not asserted (exact
  line numbers are brittle and the skill forbids pinning cosmetic formatting).
  `[CORRECTION]` the source spec's `doc0:L3` example is relaxed accordingly.
- `attributes` container: any extra keys (unknown kinds' resources, CRDs) are
  tolerated — `validate.py` ignores unknown node keys (Q1 answered). For
  contact-documented kinds, `attributes` holds `kind`, `namespace`, and
  `containers` (a list of `spec.template.spec.containers[].name`, or
  `spec.containers[].name` for Pods). Containers are attributes, not nodes.
- **Any `kind` outside an explicit list still emits a generic `K8sResource` node**
  (`label = f"{kind}/{name}"`, same id scheme) rather than being dropped (I5).
- **Embedding exclusion:** `file_type="k8s"` is *not* added to
  `embed._TEXT_FILE_TYPES` or `_EMBED_SPACE_BY_FILE_TYPE`, so K8s nodes get no
  vector — the intended deferral (§2). This is a property to keep, not a behavior
  to build.

### C4 — Edge confidence contract `[planned]` `[CORRECTION: placeholder is a node]`

| Relation | Confidence rule |
|---|---|
| `references` (ConfigMap/Secret via `envFrom`, `env[].valueFrom`, `volumes[*].configMap/secret`) | `EXTRACTED` if resolved to a node in the same namespace in the scanned corpus; `AMBIGUOUS` otherwise (target may exist only at apply time) |
| `uses_service_account` (`spec.template.spec.serviceAccountName` / `spec.serviceAccountName`) | `EXTRACTED` if resolved; `AMBIGUOUS` otherwise (`default` SA never assumed) |
| `selects` (Service → workload via `spec.selector` label match) | Always `INFERRED` — a deduction from label matching, never a stated reference, even when it resolves cleanly |

- An `AMBIGUOUS` edge still gets emitted and points at a **synthesized placeholder
  node** `k8s://{ns}/{kind}/{name}#unresolved` (`file_type="k8s"`,
  `label=f"{kind}/{name} (unresolved)"`, `attributes={"unresolved": true}`).
  `[CORRECTION]` the source spec described this as a *dangling edge target with no
  node*; `build.py:1162-1163` drops such edges, so the placeholder is a real node
  the resolver materializes in pass 2. Never silently dropped (I4).
- Every edge carries the five required fields (validate.py:7): `source`, `target`,
  `relation`, `confidence`, `source_file`.

### C5 — Two-pass extraction, resolver-registered `[planned]`

- **Pass 1 (per file):** `extract_k8s` emits nodes (C3) and stashes *candidate*
  references in a per-file side-channel key (e.g. `result["k8s_candidates"]`) —
  bare `target_name` / `target_kind` / `namespace` / `relation`, **kept out of the
  dict `validate_extraction` sees** (they are not edges yet). Mirrors how
  `swift_type_table` / `kotlin_package` side-channels carry pass-1 data into a
  resolver.
- **Pass 2 (whole corpus):** a `LanguageResolver("k8s_yaml", frozenset({".yaml",".yml"}), resolve)` registered via `register_language_resolver` at the tail of
  `extract.py` (the exact seam at extract.py:4074-4087). `resolve(per_file,
  all_nodes, all_edges)` builds a `(namespace, kind, name) -> id` index from
  `all_nodes`, resolves every candidate per C4, and appends resolved edges (and
  placeholder nodes) directly to `all_edges`/`all_nodes`.
- Ordering (verified): `run_language_resolvers` runs **inside** `extract()`
  (extract.py:6884-6892) *before* extract returns; `validate_extraction` runs only
  later at `build.py:875`. So resolver-added nodes/edges are validated at build,
  and the "dangling endpoint" error is demoted (build.py:877) while schema-field
  errors would still fire — hence pass 2 must emit all five edge fields (C4).

---

## 4. Test Targets (red-first)

One target = one behavior = one red→green. Guard targets are folded, not
serialized (see §5). Prefix `[new]` = must fail first for lack of implementation;
`[guard]` = already-true property kept locked.

### R1 targets

- **T1 `[new]` — dialect gate classifies.** `detect_dialect(path, deployments_yaml)` → `K8S_MANIFEST`; `detect_dialect(path, ci_workflow_yaml)` → `None` (no `kind`); `detect_dialect(path, argo_app_yaml)` → `None` (`argoproj.io/`); `detect_dialect(path, multi_doc_one_bad_yaml)` → `None` (C1's "every document" rule).
- **T2 `[new]` — id scheme.** A namespaced Deployment emits id `k8s://payments/Deployment/api-server`; a cluster-scoped Kind (or namespaced-with-omitted-namespace) emits `_cluster` in the namespace slot; kind/name case verbatim; the id contains `/`, `:`, `://` and no `_`-collapse.
- **T3 `[new]` — node shape + file_type.** Extracting one manifest yields a node whose id/label/source_file/source_location match C3, `file_type == "k8s"`, `attributes["containers"]` lists container names, and `attributes` carries `kind`/`namespace`.
- **T4 `[new]` — unknown kind fallback.** A CRD-shaped manifest (unknown `kind`) still emits a node (label `"{kind}/{name}"`, same id scheme) — nothing dropped (I5).
- **T5 `[new]` — routing.** Running extraction over a repo with a `.yaml` manifest produces K8s nodes (i.e. the file went code→`_DISPATCH`→`extract_k8s`); the built graph's K8s node has `file_type="k8s"`, not `"concept"` (both whitelists extended).
- **T6 `[guard]` — additive no-op.** A repo with *no* YAML/K8s file yields no `k8s`-typed nodes and no change to other extractors' output (I1). Fold into T5's slice.
- **T7 `[guard]` — determinism.** Two extraction runs over an unchanged manifest produce byte-identical node/edge dicts (I2). Fold into T5's slice.

### R2 targets

- **T8 `[new]` — resolved reference.** A Deployment referencing an in-corpus ConfigMap (same namespace) via `envFrom` yields a `references` edge with `confidence="EXTRACTED"` and `target == k8s://{ns}/ConfigMap/{name}`.
- **T9 `[new]` — unresolved reference → placeholder node.** A Deployment referencing a ConfigMap absent from the corpus yields a `references` edge with `confidence="AMBIGUOUS"` whose `target` is `k8s://{ns}/ConfigMap/{name}#unresolved`, **and** a corresponding placeholder node exists in the output (edge survives build because the target is a real node id).
- **T10 `[new]` — service account.** A Pod with `serviceAccountName` referencing an in-corpus ServiceAccount yields `uses_service_account` EXTRACTED; absent → AMBIGUOUS + placeholder. `[guard variant]` the SA name with no `metadata.namespace` on a namespaced kind resolves against `_cluster`.
- **T11 `[guard]` — confidence integrity.** Every emitted K8s edge's `confidence ∈ {EXTRACTED, INFERRED, AMBIGUOUS}` and every edge has the five required fields (validate.py:7) so no schema warning fires (I4). Fold into T8/T9's slice.

### R3 targets

- **T12 `[new]` — selects match.** A Service whose `spec.selector` labels match a Deployment's `spec.template.metadata.labels` yields a `selects` edge `Service → Deployment` with `confidence="INFERRED"` (always INFERRED even though it resolves).
- **T13 `[new]` — selects no-match / non-matching labels.** A Service whose selector matches nothing emits no `selects` edge (no false positive).

### R4 targets

- **T14 `[new]` — multi-repo composition.** Two small repos, each with a K8s manifest, run through `graphify global add` independently: every resulting id is `repo_tag::k8s://…` with the repo tag contributed **only** by `global_add`, and the two repos' ids do not collide. No extractor-side change was required (I3).

### R5 targets

- **T15 `[new]` — Neo4j label.** Pushing a K8s node produces a Neo4j node labeled `K8s` (from `file_type.capitalize()`), not `Concept`/`Entity`.
- **T16 `[guard]` — injection safety.** A K8s node/edge whose `name`/`label`/annotation value contains `'` `\` `\n` `{` `}` pushes without error and the payload stays a query parameter (never interpolated). Fold into T15's slice; this is assertion of the *existing* parameterization (graphdb.py:186-207), not new escaping.

Target count check: 7 R1 targets of which 2 are guards folded into one slice, 4
R2 targets of which 2 are guards, 2 R3, 1 R4, 2 R5 — net **~9 new-behavior cycles
across 5 requirements**, no contract above 3 *net-new* behavioral targets. Each
target asserts observable output at the `nodes`/`edges` dict or the built/global
graph boundary, never internal plumbing.

---

## 5. Escalation Map (ordered slices → §8)

Data path does NOT escalate through persistence rungs here — the "store" is
Graphify's existing `graph.json`/`global-graph.json`, already real. Escalation is
capability-by-capability along the resolver pathway:

**Slice A (R1) — "list every K8s resource as nodes."** Replace "YAML is an LLM
document" with a content-gated AST extractor. Introduce `detect_dialect` (C1),
move `.yaml`/`.yml` to code (C0), register `extract_k8s` in `_DISPATCH`, emit
nodes with the raw id scheme (C2) and `file_type="k8s"` (C3, whitelists extended).
Driving tests: T1-T5 (red, in that order) with guards T6-T7 folded in. Done when:
T1-T7 green, build produces `file_type="k8s"` nodes, non-YAML repos unchanged.

**Slice B (R2) — "resolve cross-file references."** Replace pass-1 bare candidates
with pass-2 resolution via a registered `LanguageResolver` (C5), emitting
`references`/`uses_service_account` edges (EXTRACTED/AMBIGUOUS) and materializing
placeholder nodes for unresolved targets (C4). Driving tests: T8-T10 red, T11
folded. Done when: T8-T11 green, an AMBIGUOUS edge **and its placeholder node**
survive `validate` + `build` into the graph.

**Slice C (R3) — "selects edges."** Add the label-match heuristic (Service→workload,
INFERRED) as a distinct resolver/step on the same skeleton. Driving tests: T12-T13.
Done when: both green, `selects` edges appear with `confidence="INFERRED"`.

**Slice D (R4) — "multi-repo composition."** Two-repo `global add` fixture proving
I3 with zero extractor changes. Driving test: T14. Done when: T14 green — ids are
`repo_tag::k8s://…` and the extractor contributed no repo tag.

**Slice E (R5) — "Neo4j push verification."** Confirm `K8s` labeling and
parameterized Cypher for punctuation-heavy names/annotations. Driving tests:
T15-T16. Done when: both green on a K8s node pushed to Neo4j.

No slice here is guard-only: A-E each add observable behavior. The guards (T6, T7,
T11, T16) are folded into the slice whose behavior they protect, with one
teeth-check each, not serialized as their own phases.

---

## 6. Parallelization Matrix

| Track | Slice | Safe to parallelize? |
|---|---|---|
| serial (primary) | A (R1) → B (R2) | No — B resolves the nodes A emits; B's resolver index is built from A's ids. |
| parallel | C (R3: `selects`) | Yes vs D — independent heuristic; both read A's locked node schema, neither moves it. |
| parallel | D (R4: multi-repo) | Yes vs C — pure composition fixture over A's ids; no contract it touches. |
| serial (tail) | E (R5: Neo4j) | No — depends on A's `file_type="k8s"` surviving build (A) and B's edges existing (B). |

Anything that renames a contract field, changes the id scheme, or alters the
two-pass order (`extract_k8s` → resolver) is serial by definition — it moves a
boundary C2/C5 already locked.

---

## 7. Invariants (hold across every slice)

- **I1 — Additive.** With no YAML/K8s file present, the extractor is a no-op and
  every other extractor's behavior is unaffected. *Guarded by T6.*
- **I2 — Determinism.** Two runs over an unchanged file produce byte-identical
  node/edge dicts. *Guarded by T7.*
- **I3 — No repo-identity leakage into ids.** Extractor ids never contain a repo
  tag; `global_add` is the only place one is added, uniformly as `repo_tag::`.
  *Guarded by T14 (R4).*
- **I4 — Confidence integrity.** Every K8s reference resolves to exactly one of
  `EXTRACTED`/`INFERRED`/`AMBIGUOUS`; no candidate is silently dropped (AMBIGUOUS
  → placeholder node). *Guarded by T11 (folded into R2).*
- **I5 — Unknown kinds never dropped.** Any `kind` outside the explicit list still
  yields a generic `K8sResource` node. *Guarded by T4.*
- **I6 — YAML reclassification is a visible cut, not an accident.** `.yaml`/`.yml`
  no longer produce document/semantic nodes; this regression is deliberate and
  recovers only when a future dialect slice re-adds YAML coverage. *Guarded by a
  note-level assertion in R1 (no `document`-typed node is produced from a YAML
  file after C0).*
- **I7 — `file_type="k8s"` never degrades.** It must survive `validate` + `build`
  (both whitelists) end-to-end; a K8s node that reaches the graph as `Concept`
  signals a whitelist regression. *Guarded by T5 and T15 (R5).*

---

## 8. Requirement backlog

- [ ] **R1** — Dialect detection (C1) + YAML→code routing (C0) + raw-id node
  emission (C2/C3, `file_type="k8s"` whitelisted) + dispatch wiring; generic
  fallback for unknown kinds. No edges yet. `depends: none; track: serial`
- [ ] **R2** — Two-pass resolver (C5): candidate collection + `LanguageResolver`
  registration, producing `EXTRACTED`/`AMBIGUOUS` `references` /
  `uses_service_account` edges with placeholder **nodes** for unresolved targets
  (C4). `depends: R1; track: serial`
- [ ] **R3** — `selects` edges: Service→workload via label-selector match, always
  `INFERRED` (C4). `depends: R1; track: parallel-with R4`
- [ ] **R4** — Multi-repo composition fixture: two small repos through `graphify
  global add`, verify ids compose under `repo_tag::` with zero extractor changes
  (C2/I3). `depends: R1; track: parallel-with R3`
- [ ] **R5** — Neo4j push verification for K8s nodes: `K8s` label from
  `file_type`, and confirm the existing parameterized Cypher covers
  punctuation-heavy names/labels/annotations (issue #84). No new plumbing.
  `depends: R1, R2; track: serial`

**Bootstrap note for the orchestrator.** §3 (Contracts) + §7 (Invariants) are the
always-loaded cross-cutting file; §8 is the one-at-a-time backlog. An implementer
on R1 reads §3-C0/C1/C2/C3 + §7-I1/I2/I5/I6/I7 — not R2-R5.
