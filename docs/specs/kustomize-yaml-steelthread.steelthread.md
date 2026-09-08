# Steel Thread Plan: Kustomize extractor (Graphify fork)

Third YAML dialect slice, following K8s and Argo. Reuses their seams
(`detect_dialect`, `_DISPATCH`, `extract_k8s` branch, `resolver_registry`).
Grounded against the actual Kustomize corpus under `~/Dev` on 2026-08-28.
Authority: the K8s/Argo threads' `.steelthread.md` convention.

§3 Contracts and §7 Invariants are cross-cutting. §8 is the one-at-a-time
backlog. An implementer on R1 reads §3 + §7, not R2/R3.

---

## 0. Grounding — what "Kustomize" actually is in the corpus

`kind: Kustomization` (apiVersion `kustomize.config.k8s.io/v1beta1`). Unlike K8s
and Argo, a kustomization is **not** a resource — it's a *composition recipe*:
a directory-level instruction set that (a) references other manifests by
relative path and (b) declares ConfigMaps/Secrets to be *generated* from source
files. Two distinct graph links:

1. **Composition** — `resources:` / `bases:` / `components:` (relative paths, or
   git URLs with `?ref=` — deferred) point at other manifests in the tree.
2. **Generation** — `configMapGenerator:` / `secretGenerator:` declare a
   synthesized `ConfigMap`/`Secret` whose `name` is set and whose data (`envs`,
   `files`, `literals`) is pulled at render time.

Corpus (neo4j-cloud, the only repo carrying kustomizations among the 6 selected):

| Shape | Count | Graph value |
|---|---|---|
| carries `resources`/`bases`/`components` | 861 | high — this is the composition DAG |
| carries `configMapGenerator`/`secretGenerator` | 259 | high — the artifact-generation link |
| bare or `namespace:`-only overlay | ~522 | near zero — a path-prefix convenience |

The one problem Kustomize adds over K8s/Argo is **path resolution**: a
`kustomization.yaml`'s `resources: [../base, ./api]` are *relative filesystem
paths*, not names/namespaces. So the resolver (§4 C5) is path-keyed, not
namespace/name-keyed.

---

## 1. Steel Thread Blueprint

**List every kustomization as a node; wire its `resources:`/`bases:`/`components:`
entries as `includes` edges to the manifests they compose (same tree, relative
paths resolved); and emit `generates` edges from each `configMapGenerator`/
`secretGenerator` entry to a synthesized ConfigMap/Secret node (the artifact it
produces).**

One flow, thin end to end:

1. `.yaml`/`.yml` already routes code→`_DISPATCH`→`extract_k8s`.
2. `detect_dialect` gains a `KUSTOMIZATION` dialect (C1): every doc has
   `apiVersion` starting `kustomize.config.k8s.io/` and `kind: "Kustomization"`.
3. `extract_k8s` branches to `_extract_kustomize` (C2/C3): emits one kustomization
   node, `includes` candidates (pass 1, path-keyed), and `generates` edges +
   generated-resource nodes (within-doc, resolved at extract time).
4. A `kustomize` resolver (C5) resolves `includes` candidates against the corpus —
   a kustomization's relative path is anchored at the kustomization's own file
   directory; resolve `(dir, relpath)` → that file's node id (EXTRACTED) or
   AMBIGUOUS + placeholder (dangling path).
5. Generated-resource nodes carry `file_type` = the resource kind's type policy
   (see C3 — they are themselves K8s-shaped); Neo4j labels them accordingly (I7).

The `generates` link is the capability the author flagged: it answers "where is
this ConfigMap/Secret artifact produced" — which no other dialect models.

---

## 2. Deferred (explicitly out of scope for this slice)

- `patches` / `patchesStrategicMerge` / `patchesJson6902` (transform composition —
  real but secondary; kept as attributes, not resolved edges, this slice).
- Remote `resources:` / `bases:` git URLs (`github.com/...?ref=...`) — the `?ref`
  suffix and remote fetch are deferred; only local relative paths resolve.
- `namePrefix` / `nameSuffix` / `commonLabels` / `images:` (apply-time transforms;
  captured as attributes, not wired).
- Helm chart references inside kustomizations.
- CI YAML, Makefile — their own slices.

---

## 3. Contracts

### C1 — `detect_dialect` gains `KUSTOMIZATION` `[planned]`

- `Dialect.KUSTOMIZATION` distinguishing a `kustomization.yaml` from plain K8s.
- A file classifies **KUSTOMIZATION** iff **every** top-level doc has
  `apiVersion` (str) starting with `kustomize.config.k8s.io/` **and**
  `kind == "Kustomization"`.
- Everything else unchanged: K8S_MANIFEST / ARGO_WORKFLOW / None rules as before.
- Deterministic (I2); `path` unused for the decision.

### C2 — Node id scheme (raw emission) `[planned]`

- Kustomization node: `id = f"kustomize://{dirname}/{base}"` where `dirname` is
  the kustomization's directory path (relative to the scan root, `/`-joined,
  verbatim) and `base` = the filename (`kustomization.yaml` / `kustomization.yml`).
  `dirname` is the CORPUS-RELATIVE directory (no `/Users/...` absolute path; no
  repo identity — I3). Example: `kustomize://components/gds-api/k8s/base/kustomization.yaml`.
- Generated-resource node: `id = f"k8s://{namespace}/{kind}/{name}"` with the SAME
  `k8s://` scheme K8s extraction uses (a `configMapGenerator` yields a ConfigMap
  with that exact id) — so a generated ConfigMap is addressable by the K8s
  reference resolver when a workload later envFrom-s it. `namespace` from the
  generator `namespace:` key, or `_cluster` if absent.
- Raw verbatim (never `ids.make_id`/`normalize_id`); case-preserved.

### C3 — Node shape `[planned]`

Kustomization node:
```json
{
  "id": "kustomize://components/gds-api/k8s/base/kustomization.yaml",
  "label": "kustomization.yaml",
  "file_type": "kustomize",
  "source_file": "components/gds-api/k8s/base/kustomization.yaml",
  "source_location": "doc0",
  "attributes": {
    "dir": "components/gds-api/k8s/base",
    "resources": ["../base", "./api"],
    "namespace": "gds-api"
  }
}
```
Generated-resource node (`configMapGenerator`/`secretGenerator` entry):
```json
{
  "id": "k8s://_cluster/ConfigMap/gds-api-session-sizing",
  "label": "ConfigMap/gds-api-session-sizing",
  "file_type": "k8s",
  "source_file": "components/gds-api/k8s/overlays/brain/kustomization.yaml",
  "source_location": "doc0",
  "attributes": {"kind": "ConfigMap", "generated_by": "kustomize://…/kustomization.yaml", "generator": "configMapGenerator", "name": "gds-api-session-sizing"}
}
```
- `file_type="kustomize"` for the kustomization node (added to both whitelists,
  I7). Generated-resource nodes use `file_type="k8s"` (they ARE K8s resources,
  so `"k8s"` is already whitelisted — no new type).
- `resources`/`bases`/`components` entries are carried as an attributes list
  (verbatim), AND emitted as `includes` candidates/edges per C4.
- `namespace`/`commonLabels`/`namePrefix`/`nameSuffix` are attributes (not wired).

### C4 — Edge contracts `[planned]`

| Relation | Source → Target | Confidence rule |
|---|---|---|
| `includes` | kustomization node → the node for a manifest it lists in `resources`/`bases`/`components` | `EXTRACTED` when the relative path resolves to a file that produced a node in the corpus; `AMBIGUOUS` (+ placeholder) otherwise |
| `generates` | kustomization node → the synthesized ConfigMap/Secret node | Always `EXTRACTED` — the target is the kustomization's own declaration, not a lookup |

- `generates` is resolved within-doc at extract time (the generated node id is
  deterministically derived from the generator's `name` + `kind` + `namespace`).
- `includes` is cross-file (path-keyed), resolved in pass 2 (C5).
- `AMBIGUOUS` `includes` targets a synthesized placeholder node
  `kustomize://{dir}/{base}#unresolved` (a path placeholder), never a dangle
  (build.py drops dangles). `[CORRECTION carried from K8s C4.]`
- Every edge carries the five required fields (validate.py:7).

### C5 — Two-pass, resolver-registered `[planned]`

- **Pass 1** (`_extract_kustomize`): emit the kustomization node + generated-resource
  nodes + `generates` edges; stash `includes` candidates in a per-file
  `kustomize_candidates` side-channel: `{source, target_path, dir, source_file,
  relation="includes"}` where `target_path` is the raw relative path from the
  kustomization's `resources:`/`bases:`/`components:` list and `dir` is the
  kustomization's directory (its resolution anchor).
- **Pass 2**: a `LanguageResolver("kustomize", frozenset({".yaml",".yml"}),
  resolve)` builds an index of `(source_dir, file_basename) -> node_id` from every
  kustomization/`k8s://` node's `source_file`, resolves each `includes` candidate by
  joining `dir` + relative path and normalizing (`posixpath.normpath`), emits
  EXTRACTED/AMBIGUOUS edges, and synthesizes placeholder nodes for unresolved paths.
- Resolution anchor rule: a kustomization's `resources:`/`bases:`/`components:`
  paths are relative to that kustomization's own directory (Kustomize semantics).
  `../` is legal. Remote URLs (containing `://` or starting `http`/`git`) are
  skipped (deferred, §2).

---

## 4. Test Targets (red-first)

### R1 (detection + extraction + generates)
- **T1** — `detect_dialect` classifies a `kustomization.yaml` (`apiVersion:
  kustomize.config.k8s.io/v1beta1`, `kind: Kustomization`) as `KUSTOMIZATION`;
  a plain K8s ConfigMap and an Argo Workflow still classify as their own dialect.
- **T2** — kustomization node id `kustomize://components/gds-api/k8s/base/kustomization.yaml`
  (dir relative to scan root, filename included); raw, verbatim.
- **T3** — a kustomization with `resources: [../base, ./api]` carries those in
  `attributes["resources"]`, and the kustomization node has `file_type="kustomize"`.
- **T4** — `generates`: a `configMapGenerator` entry emits a ConfigMap node
  `k8s://_cluster/ConfigMap/<name>` (namespace from the generator or `_cluster`) and
  a `generates` edge `kustomization → that node` with `confidence="EXTRACTED"`.
- **T5** — `secretGenerator` likewise yields a Secret node (`target_kind`/kind
  "Secret").

### R2 (includes resolution)
- **T6** — `includes` EXTRACTED: two kustomizations, `A` with `resources: [../base]`
  and `B` at that resolved path, yields an `includes` edge `A → B` EXTRACTED.
- **T7** — `includes` AMBIGUOUS: a `resources: [./missing.yaml]` with no such file
  yields an AMBIGUOUS edge + a `kustomize://…/missing.yaml#unresolved` placeholder node.

### R3 (whitelist/Neo4j + determinism)
- **T8** — `file_type="kustomize"` passes validation and survives build as
  "kustomize".
- **T9** — determinism: two runs over an unchanged kustomization are equal (I2).
- **T10** — Neo4j: a kustomization node pushes with label `Kustomize` (I7); a
  generated-resource (`k8s`) node pushes with label `K8s` (already shipped, re-pin
  via path).

---

## 5. Escalation Map (ordered slices → §8)

1. **R1** — dialect + kustomization-node emission + `generates` edges + generated
   resource nodes. Driving tests T1-T5.
2. **R2** — path-keyed `includes` resolver (C5). Driving tests T6-T7.
3. **R3** — `file_type="kustomize"` whitelist + determinism + Neo4j label pins.
   Driving tests T8-T10.

Each slice adds observable behavior; none is guard-only.

---

## 6. Parallelization Matrix

| Track | Slice | Parallelizable? |
|---|---|---|
| serial | R1 → R2 | No — R2 resolves the nodes R1 emits (generated-resource nodes are also the `k8s://` index targets). |
| serial | R2 → R3 | No — R3 pushes/asserts the graph R1/R2 produce. |

Serial.

---

## 7. Invariants (hold across every slice)

- **I1 — Additive.** A repo with no kustomization is a no-op; K8s/Argo paths unchanged.
- **I2 — Determinism.** Byte-identical output across runs.
- **I3 — No repo identity in ids.** `kustomize://` and `k8s://` ids never contain a
  repo tag; `global_add` alone adds `repo_tag::`.
- **I4 — Confidence integrity.** Every `includes`/`generates` edge resolves to
  EXTRACTED/AMBIGUOUS; nothing silently dropped.
- **I5 — Generated resources are K8s-addressable.** A generated ConfigMap/Secret
  id uses the `k8s://` scheme so the existing K8s `references` resolver can target it.
- **I6 — Kustomization ≠ resource.** A `kind: Kustomization` does not emit a
  plain-K8s node (it is a composition recipe, modeled by `kustomize://`).
- **I7 — `file_type`s never degrade.** `kustomize` survives validate+build and
  pushes as `Kustomize`; generated `k8s` nodes push as `K8s`.

---

## 8. Requirement backlog

- [ ] **R1** — Kustomize dialect detection (C1) + kustomization node emission (C2/C3)
  + `generates` edges + generated-resource nodes (C4). `depends: none; track: serial`
- [ ] **R2** — Path-keyed `includes` resolver (C5): EXTRACTED/AMBIGUOUS with placeholder
  nodes. `depends: R1; track: serial`
- [ ] **R3** — `file_type="kustomize"` whitelist + determinism + Neo4j `Kustomize`
  label pin (I7). `depends: R1, R2; track: serial`

**Bootstrap note:** §3 + §7 are the always-loaded cross-cutting file; §8 the
one-at-a-time backlog.
