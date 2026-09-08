# Steel Thread Plan: Argo Workflows extractor (Graphify fork)

Supersedes nothing — this is the second YAML dialect slice, following the K8s
thread. It reuses that thread's seams (`detect_dialect`, `_DISPATCH`, the
`k8s_yaml` extractor entry point, `resolver_registry`) and deliberately adds no
new plumbing. Authority: the K8s thread's `.steelthread.md` convention, grounded
against the actual Argo corpus in `~/Dev` on 2026-08-27.

§3 Contracts and §7 Invariants are cross-cutting — read on every requirement's
plan step. §8 is the one-at-a-time backlog an orchestrator consumes.

---

## 0. Grounding — what the corpus actually contains

Scoped to the six repos the author selected (kg-builder, genai-cloud,
text2cypher, neo4j-graphrag-python, upx, neo4j-cloud). "Argo" here means the
`argoproj.io/v1alpha1` workflow kinds only — `Workflow`, `WorkflowTemplate`,
`ClusterWorkflowTemplate`, `CronWorkflow`. Counts are real-workflow files, not
every file that *mentions* `argoproj.io` (RBAC/storage/IAM files mention the API
group but are plain K8s resources and stay on the K8s dialect).

| Repo | Real Argo workflows | Character |
|---|---|---|
| genai-cloud | 4 | hera-generated dbt-build WorkflowTemplates (`DO NOT EDIT`) |
| kg-builder | 1 | hand-written `hello-artifacts-wftmpl.yaml` — the canonical DAG exemplar |
| neo4j-cloud | ~36 | the bulk: aura-operations (13), billed-usage-exporter, cloud-costs-api (incl. a CronWorkflow), quality-tools, speedy-importer, GDS benchmarking |
| text2cypher | 0 | its `local-all-in-one.yaml` is plain multi-doc K8s (matched `argoproj.io` only via an RBAC subject) |
| upx, neo4j-graphrag-python | 0 | none |

Structural surface observed:

- **Two execution shapes** inside `spec.templates[]`: `dag.tasks[]` (with
  `dependencies: [...]` and `template:` refs) and `steps[][]` (nested lists, each
  step a `template:` ref). Both reference a *named template* in the same doc's
  `templates` list.
- **Cross-file reference**: `spec.workflowSpec.workflowTemplateRef.name` (a
  CronWorkflow / Workflow referencing a `WorkflowTemplate` by name), and the
  cluster-scoped `clusterWorkflowTemplateRef` → `ClusterWorkflowTemplate`.
- **Parameters/artifacts** (`{{workflow.parameters.X}}`,
  `{{tasks.produce.outputs.artifacts.message}}`) are string interpolation inside
  `container.args`/`arguments` — they ride *inside* existing scalar attribute
  values and are **not** independently resolved in this slice (deferred, §2).
- **Non-goal noise**: `workflow-template-image-path.yaml` files (×16) are a bare
  `images: [{path, kind}]` schema-patch list with no `apiVersion`/`kind`. They
  fall through `detect_dialect` as non-manifest and are ignored (I1).

The one hard problem this dialect adds over K8s: **a second node granularity.**
Templates are addressed *by name* across the DAG and (via `workflowTemplateRef`)
across files, so unlike K8s containers they must be nodes, not attributes.

---

## 1. Steel Thread Blueprint

**List every Argo Workflow/WorkflowTemplate/CronWorkflow as a graph node; its
`templates[]` as child template nodes; wire the DAG (`template:` refs as
`invokes`, `dag.tasks[].dependencies` as `depends_on`); and resolve cross-file
`workflowTemplateRef` references with confidence labels.**

One flow, thin end to end:

1. `.yaml`/`.yml` is already routed code→`_DISPATCH`→`extract_k8s` (K8s thread C0).
2. `detect_dialect` gains an `ARGO_WORKFLOW` dialect (C1): a file classifies when
   every top-level doc has `apiVersion` starting `argoproj.io/` and a `kind`
   in the four workflow kinds. Everything else falls through unchanged.
3. `extract_k8s` branches on the dialect: K8s docs → existing K8s path (unchanged);
   Argo docs → the new `extract_argo` path (C2/C3) emitting workflow + template
   nodes and the within-doc `invokes`/`depends_on` edges, stashing
   `workflowTemplateRef` candidates for pass 2.
4. A new `argo` resolver (C4/C5) runs pass 2, resolving `workflowTemplateRef`
   names against the `(namespace, WorkflowTemplate, name)` index → `references`
   edges (EXTRACTED/AMBIGUOUS), synthesizing a placeholder node when unresolved.
5. `file_type="argo"` joins the two whitelists (C3) so argo nodes survive
   validate+build, and the Neo4j push labels them `Argo` (I7).

Permanent non-goals (§2) are unchanged: Helm, Kustomize, CI YAML, Makefile are
each their OWN future slice — Argo does not absorb them.

---

## 2. Deferred (explicitly out of scope for this slice)

- Helm template rendering. (unchanged from K8s thread)
- Kustomize overlay resolution. (its own slice, after Argo)
- CI YAML (GitHub Actions / GitLab). (its own slice)
- The neo4j-cloud Makefile dialect. (its own slice; not even YAML)
- Argo **parameter/artifact interpolation** — resolving `{{inputs.parameters.X}}`
  / `{{workflow.parameters.X}}` / `{{steps.X.outputs...}}` into explicit
  data-flow edges. The strings today ride verbatim inside `attributes`; the
  parameter graph is deferred (it would double the edge surface for a
  third-pass resolution). *Flagged, not dropped — see §7 open note.*
- `workflow-template-image-path.yaml` schema-patch files (not workflows, ignored).

---

## 3. Contracts

### C1 — `detect_dialect(path, raw_text) -> Dialect | None` extended `[planned]`

`Dialect` gains `ARGO_WORKFLOW` (today it has only `K8S_MANIFEST`).

- A file classifies **ARGO_WORKFLOW** iff **every** top-level document has
  `apiVersion` (str) beginning with `argoproj.io/` **and** `kind` (str) in
  `{"Workflow", "WorkflowTemplate", "ClusterWorkflowTemplate", "CronWorkflow"}`.
- Everything else is unchanged: `K8S_MANIFEST` when every doc has `apiVersion`+
  `kind` and none is `argoproj.io/`-prefixed; `None` for empty/unparseable/
  non-mapping/missing-field.
- A mixed file (one Argo doc + one plain K8s doc) returns `None` — the "every
  document" rule means no document is half-classified. `[CORRECTION from K8s
  thread: the K8s C1 already reserved the argoproj.io/ prefix by returning None;
  this slice replaces that None with the ARGO_WORKFLOW branch.]`
- Deterministic (I2); `path` unused for the decision.

### C2 — Node id scheme (raw emission, two levels) `[planned]`

Same verbatim/raw rule as K8s C2 (never through `ids.make_id`/`normalize_id`):

- Workflow-level node: `id = f"argo://{namespace}/{kind}/{name}"` where `kind` is
  one of the four workflow kinds and `name = metadata.name`. `namespace =
  metadata.namespace` else `"_cluster"` (a `ClusterWorkflowTemplate`/cluster-scoped
  resource simply has no namespace, so it is `_cluster` automatically).
- Template node: `id = f"argo://{namespace}/{kind}/{name}/{template}"` where
  `template` is the `name` field of the `spec.templates[]` entry. Template names
  are DNS-safe (no `/`), so the 4-segment split (`split("/", 3)`) is unambiguous.
- Kind/name/template strings are case-preserved (K8s thread C2 rationale).
- No repo identity in ids (I3); `global_add` composes them under `repo_tag::`.

### C3 — Node shape `[planned]`

Workflow-level node:
```json
{
  "id": "argo://kg-builder/WorkflowTemplate/hello-artifacts",
  "label": "WorkflowTemplate/hello-artifacts",
  "file_type": "argo",
  "source_file": "workflows/definitions/examples/hello-artifacts-wftmpl.yaml",
  "source_location": "doc0",
  "attributes": {"kind": "WorkflowTemplate", "namespace": "kg-builder",
                 "entrypoint": "main"}
}
```
Template node:
```json
{
  "id": "argo://kg-builder/WorkflowTemplate/hello-artifacts/produce",
  "label": "produce",
  "file_type": "argo",
  "source_file": "workflows/definitions/examples/hello-artifacts-wftmpl.yaml",
  "source_location": "doc0",
  "attributes": {"template": "produce", "parent": "hello-artifacts",
                 "container_image": "alpine:3.20"}
}
```
- `file_type="argo"` — added to `validate.py:4` `VALID_FILE_TYPES` and `build.py`
  inline allowed-set (same I7 mechanism as the K8s `"k8s"`).
- Template nodes are **nodes**, not attributes — they are the cross-reference
  target (unlike K8s containers, which remain attributes).
- `entrypoint` captured when present (`workflow`/workflow-level `spec.entrypoint`).
- `container_image` captured when the template has `container: {image: ...}` or
  `script: {image: ...}`; absent otherwise (present/non-empty only).

### C4 — Edge contracts `[planned]`

| Relation | Source → Target | Confidence rule |
|---|---|---|
| `invokes` | containing template node → referenced template node (from a `dag.tasks[].template` or `steps[][]` step's `template:` ref) | `EXTRACTED` when the referenced name resolves to a `templates[]` entry in the same document; `AMBIGUOUS` (+ placeholder node) otherwise (a dangling template ref) |
| `depends_on` | the template a dependent task invokes → the template its `dependencies[]` tasks invoke | `INFERRED` — a deduction from task-level ordering collapsed onto templates, never a stated template→template edge |
| `references` | workflow-level node → `WorkflowTemplate`/`ClusterWorkflowTemplate` node (from `spec.workflowSpec.workflowTemplateRef.name` / `clusterWorkflowTemplateRef.name`) | `EXTRACTED` when a template of that name resolves in the scanned corpus (same namespace for namespaced refs, `_cluster` for cluster refs); `AMBIGUOUS` (+ placeholder) otherwise |

- `invokes` and `depends_on` are within-document, resolved in pass 1 (the whole
  `templates[]` list is visible at extract time).
- `references` is cross-file, resolved in pass 2 (C5) — the C4-analog capability.
- An `AMBIGUOUS` edge always points at a synthesized placeholder **node**
  (id `f"argo://{ns}/{kind}/{name}#unresolved"` or the template form
  `f"argo://{ns}/{kind}/{name}/{template}#unresolved"`), never a dangling edge
  (build.py:1162 drops those). `[CORRECTION carried from K8s C4.]`
- Every edge carries the five required fields (validate.py:7): `source`, `target`,
  `relation`, `confidence`, `source_file`.

### C5 — Two-pass, resolver-registered `[planned]`

- **Pass 1** (`extract_argo`, inside the `.yaml` entry point): emit workflow +
  template nodes (C3), emit `invokes`/`depends_on` edges (within-doc, C4), and
  stash `workflowTemplateRef`/`clusterWorkflowTemplateRef` as bare candidates in
  the per-file `argo_candidates` side-channel (`{source_id, target_name,
  target_kind, namespace, source_file, relation="references"}`) — kept out of the
  dict validation sees (mirrors `k8s_candidates`).
- **Pass 2**: a `LanguageResolver("argo", frozenset({".yaml",".yml"}), resolve)`
  registered via `register_language_resolver`; builds a
  `(namespace, WorkflowTemplate|ClusterWorkflowTemplate, name) -> id` index from
  `all_nodes`, resolves each candidate per C4, appends `references` edges +
  placeholder nodes.
- Ordering (verified): resolvers run inside `extract()` before build-time
  validation; the argo resolver runs alongside `k8s_yaml` (both `.yaml`-gated,
  both additive — neither reads the other's edges).

---

## 4. Test Targets (red-first)

Targets are %22new%22 (fail-first) unless marked `[guard]`.

### R1 targets (dialect + nodes + within-doc edges)

- **T1 `[new]`** — `detect_dialect` classifies a `WorkflowTemplate`/`Workflow`/
  `CronWorkflow` doc as `ARGO_WORKFLOW`; a mixed Argo+K8s file, a `job.yaml`
  (batch/v1), and a `workflow-template-image-path.yaml` all stay `None`/`K8S_MANIFEST`.
- **T2 `[new]`** — id scheme: workflow node `argo://kg-builder/WorkflowTemplate/
  hello-artifacts`; template node `argo://kg-builder/WorkflowTemplate/hello-artifacts/produce`;
  a `ClusterWorkflowTemplate` (no namespace) yields `_cluster`.
- **T3 `[new]`** — a `dag` WorkflowTemplate emits one workflow node + N template
  nodes (one per `templates[]`), each with correct label/file_type/source_file and
  attributes (`entrypoint`, `container_image`).
- **T4 `[new]`** — `invokes`: a `dag.tasks[]` entry with `template: produce`
  yields an `invokes` edge `main → produce` with `confidence="EXTRACTED"`.
- **T5 `[new]`** — `depends_on`: `dependencies: [produce]` on task `consume`
  yields a `depends_on` edge `consume-template → produce-template` with
  `confidence="INFERRED"`.
- **T6 `[guard]`** — a non-Argo repo is a no-op: an Argo-free YAML file emits no
  `argo`-typed nodes and no argo edges (I1, I1-additive). Fold into T3's slice.

### R2 targets (cross-file `references`)

- **T7 `[new]`** — `references`: a `CronWorkflow` with
  `workflowTemplateRef.name: daily-env-cost-ingestion` + a matching
  `WorkflowTemplate` in the corpus yields a `references` edge
  `CronWorkflow → WorkflowTemplate` with `confidence="EXTRACTED"`.
- **T8 `[new]`** — unresolved `workflowTemplateRef` yields `AMBIGUOUS` + a
  placeholder **node** `argo://{ns}/WorkflowTemplate/{name}#unresolved` (edge
  survives build).
- **T9 `[guard]`** — confidence integrity: every emitted argo edge has
  `confidence ∈ {EXTRACTED, INFERRED, AMBIGUOUS}` and five required fields
  (I4). Fold into T7/T8's slice.

### R3 targets (Neo4j + integrity)

- **T10 `[new]`** — a `file_type="argo"` node pushes with an `Argo` label token
  (mirrors the K8s R5 `K8s` label pin).
- **T11 `[guard]`** — determinism: two runs over an unchanged workflow produce
  byte-identical node/edge dicts (I2). Fold into R1's final slice.

---

## 5. Escalation Map (ordered slices → §8)

- **Slice A (R1)** — dialect branch + workflow/template node emission +
  within-doc `invokes`/`depends_on` edges + `file_type="argo"` whitelists.
  Driving tests T1–T5 (T6 folded).
- **Slice B (R2)** — cross-file `workflowTemplateRef` resolver (C5) emitting
  `references` EXTRACTED/AMBIGUOUS + placeholder nodes. Driving tests T7–T8 (T9
  folded).
- **Slice C (R3)** — Neo4j `Argo` label pin + determinism guard. Driving tests
  T10 (T11 folded).

No slice is guard-only; each adds observable behavior.

---

## 6. Parallelization Matrix

| Track | Slice | Parallelizable? |
|---|---|---|
| serial | A (R1) → B (R2) | No — B resolves the template nodes A emits. |
| serial | B → C (R3) | No — C pushes a graph that B's nodes/edges feed. |

Serial. (Nothing here branches on independent boundaries as with K8s R3/R4; the
multi-repo composition is already covered generically by `global_add`.)

---

## 7. Invariants (hold across every slice)

- **I1 — Additive.** With no Argo file present, the extractor is a no-op; a plain
  K8s manifest still runs the K8s path unchanged. *Guarded by T6.*
- **I2 — Determinism.** Two runs over an unchanged file → byte-identical dicts.
  *Guarded by T11.*
- **I3 — No repo identity in ids.** Extractor ids never contain a repo tag;
  `global_add` alone adds `repo_tag::`. *Inherited from K8s R4 (already proven
  generically for any `//…`-shaped id); not re-tested here.*
- **I4 — Confidence integrity.** Every argo edge resolves to exactly
  EXTRACTED/INFERRED/AMBIGUOUS; no reference silently dropped (AMBIGUOUS →
  placeholder node). *Guarded by T9.*
- **I5 — Unknown workflow kinds never dropped.** A `kind` outside the four
  workflow kinds but with `argoproj.io/` apiVersion (e.g. an `argoproj.io` CRD we
  don't model) still classifies as ARGO_WORKFLOW and emits a workflow-level node
  with a generic shape, or falls through to K8s if it carries a plain
  `apiVersion`+`kind`. *No silent drop.*
- **I6 — `argoproj.io/` files are no longer "not a manifest".** Before this slice,
  `detect_dialect` returned `None` for them; now they classify. No plain-K8s file
  regresses.
- **I7 — `file_type="argo"` never degrades.** It survives validate+build (both
  whitelists) and Neo4j pushes as `Argo`. *Guarded by T10.*

**Open note (non-blocking):** parameter/artifact data-flow edges are deferred
(§2). When a later slice adds them, it will extend C4 with a new relation — do
not fold parameter resolution into the `invokes`/`depends_on` semantics.

---

## 8. Requirement backlog

- [ ] **R1** — Argo dialect detection (C1) + workflow/template node emission (C2/C3,
  `file_type="argo"` whitelisted) + within-doc `invokes`/`depends_on` edges (C4).
  `depends: none; track: serial`
- [ ] **R2** — Cross-file `workflowTemplateRef`/`clusterWorkflowTemplateRef` resolver
  (C5): `references` edges EXTRACTED/AMBIGUOUS with placeholder **nodes**. `depends:
  R1; track: serial`
- [ ] **R3** — Neo4j `Argo` label pin (I7) + determinism guard (I2). No new
  plumbing. `depends: R1, R2; track: serial`

**Bootstrap note for the orchestrator.** §3 (Contracts) + §7 (Invariants) are the
always-loaded cross-cutting file; §8 the one-at-a-time backlog. An implementer on
R1 reads §3-C1/C2/C3/C4 + §7-I1/I5/I6/I7, not R2/R3.
