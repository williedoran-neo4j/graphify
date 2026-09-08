# Steel Thread Plan: Cross-repo "uber graph" (build→deploy edge stitching)

Supersedes the informal "super-graph" milestone note as the authoritative implementer
spec. Produced by the steel-thread skill, grounded against the actual `~/Dev` corpus on
2026-08-30. Where an assumption contradicts the code, it is recorded as a `[CORRECTION]`.

The thesis in one line: **an image (or more generally an artifact) is the shared join
key that turns N separately-indexed repo graphs into one traversable graph.** A
Deployment in repo A `runs` image X; a CI job in repo B `publishes` image X; a Makefile
in repo C `builds` image X. If image X is emitted as a node whose identity is its
*canonical registry path* (not its file, not its repo) and whose `source_file` is `None`,
then `global_add`'s existing external-node label-merge collapses all three repos' copies
of X into **one node** — and the `runs` / `publishes` / `builds` edges that already point at
each repo's copy are rewired onto that one node. The result is a real cross-repo traversal
(Deploy → Image ← CI → Makefile) with **zero new global-graph plumbing**: the links are
semantic edges, not `repo_tag::` co-location.

Everything else in this document layers that one mechanism across new dialects (CI YAML,
Makefile/Dockerfile/.mk) and a backfill (k8s `image:` capture).

---

## 0. Grounding — what the corpus and code actually contain

Target repos (first slice; from `~/Dev`, 13 git repos surveyed):

| Repo | Role | GH Actions workflows | Makefiles/`.mk` | Concrete images |
|---|---|---|---|---|
| `neo4j-cloud` | deployment: Go services + pointers to non-Go components | 815 | 353 | `europe-west1-docker.pkg.dev/aura-docker-images/aura/**` |
| `kg-builder` | uses `neo4j-graphrag` (source-swapped) | ~12 | — | `europe-west1-docker.pkg.dev/genai-aura-5bf9/kg-builder/**` |
| `upx` | frontend | 94 | — | `quay.io/keycloak/**`, `neo4j:*-enterprise`, local `node` |
| `text2cypher` | component | 7 | — | — |

`genai-cloud` is evaluation infrastructure — **out of scope** for the thread (§2).
`neo4j-graphrag-python` is a *library dependency* of `kg-builder`, joined by code identity,
not image identity — flagged as the §2 deferred "cross-repo component (library) link".

Five code facts that shaped the plan, all verified against source (not assumed):

1. **`.yaml`/`.yml` already route to `extract_k8s`** (`extract.py:_DISPATCH`, lines 5262-5263),
   but `extract_k8s`→`detect_dialect` returns `None` for non-k8s/argo/kustomize YAML
   (`k8s.py:60`) — so a GitHub Actions workflow is a **silent no-op today**: no node, no error.
2. **`Makefile` / `Dockerfile` are extensionless** and `.mk` is *not* in `CODE_EXTENSIONS`
   (`detect.py:44`), so all three fall into the "unclassified" bucket (`detect.py:1889`) and
   leave **no trace** — not counted, not listed.
3. **The k8s extractor captures container *names* (`_container_names`, k8s.py:88) but NOT
   `image:` refs.** Argo's extractor captures `container_image` (`k8s.py:276-277`); k8s does not.
   This is the missing "consumer" leg — a Deployment doesn't currently know what image it runs.
4. **Corpus images mix concrete registry paths and placeholder vars.** Real examples:
   `europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility:806406f5…-unclean`,
   `…/aura-operations-utility@sha256:5eb01a63…`, `quay.io/keycloak/keycloak:19.0.3` — beside
   `DEV_IMAGE`, `_IMAGE`, `_NEO4J_IMAGE_ID`, `gds-api-image`, `ko://node-scaler`. Image capture
   must therefore *classify*: concrete registry path → image node; placeholder/var/`ko://` → skip.
5. **`global_add` already merges `source_file`-falsy nodes by `label`** (`global_graph.py:123-134`),
   rewiring their incident edges (`remap`). This is the pre-existing seam the cross-repo join rides
   on. `validate.py` requires the `source_file` *key* to be present but does not check truthiness
   (`REQUIRED_NODE_FIELDS`, validate.py:6, checked with `field not in node`) — so `source_file: None`
   is valid and satisfies both validation and the external-merge trigger.

`[CORRECTION]` to the natal assumption "build→deploy cross-repo links exist today": in the real
corpus, `neo4j-cloud` and `kg-builder` publish to *different* registries (`aura-docker-images` vs
`genai-aura-5bf9`), so no single image is genuinely built in one repo and deployed in another
*within this slice's repos*. The thread therefore proves the mechanism with a **synthetic two-repo
fixture** (the established pattern — cf. the k8s R4/T14 "two small repos" fixture), while the real
corpus provides the *within-repo* build→deploy richness (`neo4j-cloud`'s 353 Makefiles + 815
workflows + its k8s manifests all sharing `aura-docker-images`). The cross-repo *component* link
that already exists in the corpus (`kg-builder` depends on `neo4j-graphrag`) is a different join
key (library identity) and is deferred to §2.

---

## 1. Steel Thread Blueprint

The one transaction: **given an image reference, traverse from what deploys it to what builds it,
across repos.**

Thin end-to-end shape (all hardcoded where the dialect real logic is not yet written):

1. Two small synthetic repos — `builder/` (mirrors kg-builder) and `cloud/` (mirrors neo4j-cloud) —
   are each run through the extractor.
2. `builder/` has a `Dockerfile` (`FROM` → image node), a Makefile target that names the image, and
   a `.github/workflows/build-and-publish.yml` with a `docker build/push` step → these three each
   emit an edge to the same image node `image://europe-west1-docker.pkg.dev/genai-aura-5bf9/kg-builder/schema`.
3. `cloud/` has a k8s `Deployment` whose container `image:` is that same registry path → it emits a
   `runs` edge to the same image node.
4. `graphify global add` is run for `builder` then `cloud`. Because the image node carries
   `source_file: None` and `label == "europe-west1-docker.pkg.dev/genai-aura-5bf9/kg-builder/schema"`
   (the full canonical path — the merge key), the two repos' image nodes collapse onto one node.
5. The merged graph now contains, over **real edges**: `Deployment --runs--> Image <-publishes-- CI`
   and `Makefile --builds--> Image` — a genuine cross-repo traversal, no repo inside any id, the
   `repo_tag::` prefix supplied only by `global_add` at merge time.

What this proves: the shape works end-to-end — *extract producer legs → common image identity →
consumer leg → cross-repo merge → navigable graph* — before any dialect is made fully real.

Deliberately hardcoded in the thread: each dialect extractor returns a fixed single-node/edge
fixture rather than parsing real YAML/Makefile bodies. That hardcoding is the *green* step of the
first slice; every dialect then replaces it with real parsing in its own escalation slice.

---

## 2. Deferred (explicitly out of scope for the thread)

- **`genai-cloud` (eval infra)** — not in the thread's corpus; add later verbatim.
- **Cross-repo *component* (library) link** — `kg-builder` → `neo4j-graphrag-python` (code identity,
  not image identity). Separate join key; its own slice after images.
- **GitLab CI** — GitHub Actions first (the corpus is GH Actions); GitLab is the same shape, a later dialect.
- **docker-compose** — lower priority; the corpus has one compose file. Same image-join mechanism, its own slice.
- **Ansible** — named in the earlier specs as a future slice; no corpus presence in this repo set, so deferred.
- **Helm** — template *rendering* (unrendered `{{ }}` masks parsing); chart *structure* referencing image
  values is in scope only insofar as a plain k8s manifest path already handles it. Full Helm = its own slice.
- **Argo parameter/artifact interpolation** — resolving `{{inputs.parameters.X}}` into data-flow edges
  (carried verbatim today). Independent of the image join.
- **`ko://` / build-var image resolution** — resolving `DEV_IMAGE`, `_IMAGE`, `_NEO4J_IMAGE_ID`,
  `ko://node-scaler` placeholders to concrete registries (would require build-var plumbing). In the
  thread they are **skipped**, not resolved (see I5).
- **Image *tag* semantics** — semantic-version / digest-alias reasoning beyond "tag is an attribute".
  Identity is the tag-less registry path (§3 C1); no tag ordering/channel logic.

---

## 3. Contracts

### C1 — Image node (the join key)

- `id = f"image://{registry_path}"` where `registry_path` is the concrete image reference with the
  **tag and digest stripped** (the case-preserved `registry[:port]/[ns/…/]name`, no trailing `:`).
  Examples:
  - `europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility`
  - `quay.io/keycloak/keycloak`
- `label` = the same full `registry_path` (this is the `global_add` merge key — §1).
- `file_type = "image"` (new; added to both whitelists — validate.py:4 and build.py:856).
- `source_file = None` (triggers `global_add` external-merge; images have no owning file).
- `attributes = {"registry": <authority, e.g. "europe-west1-docker.pkg.dev">, "tags": [<verbatim tag or digest strings>]}`.
  A `@sha256:…` digest is captured as a tag string too, not as identity.
- Raw verbatim id (never `ids.make_id`/`normalize_id`); case is preserved.

### C2 — k8s `image:` capture (the consumer leg)

- The k8s extractor (`_extract_k8s`, k8s.py) additionally reads each container's `image:` string
  (Deployment-like and Pod specs — the same specs `_container_names` already walks).
- Concrete registry path → emit the image node (C1) *and* a `runs` edge:
  `{source: <the workload node id>, target: <image node id>, relation: "runs",
  confidence: "EXTRACTED", source_file: <the manifest path>}`.
- A `runs` edge is emitted per distinct workload→image pair. The five required edge fields hold.
- `image:` values that are placeholders/vars/`ko://` (`DEV_IMAGE`, `_IMAGE`, `_NEO4J_IMAGE_ID`,
  `ko://…`, any value without a `/` registry separator, or a bare name) are **skipped** (no image node,
  no edge). See I5.

### C3 — Makefile / `.mk` / Dockerfile extractor (the build leg)

- New `file_type = "build"` (whitelisted) for the *target/stage* node; images stay `image` (C1).
- **Routing (new, the load-bearing move):** `Makefile`, `Dockerfile`, and any `.mk` file must be
  classified into the code path. `Makefile`/`Dockerfile` are extensionless; `.mk` is not a code
  extension today. The thread adds them to `detect.py`/`_DISPATCH` so a new `extract_makefile`
  (or `_extract_build`) runs on them rather than dropping them as "unclassified".
- **Makefile/`.mk` node:** one node per named target, `id = f"makefile://{package_rel_dir}/{target}"`
  semantics (exact scheme decided at implement time, but stable + `/`-joined + no repo identity,
  mirroring `kustomize://`), `label = <target>`, `file_type = "build"`, `source_file = str(path)`.
  Targets that name a concrete image in their recipe → `builds` edge to that image
  `{relation: "builds", confidence: "EXTRACTED"}`.
- **Dockerfile node:** one node per `FROM` stage image, `file_type = "build"` for the *stage* node,
  `source_file = str(path)`; the `FROM <image>` → `builds` edge to the image node (C1), EXTRACTED.
- The thread's green step hardcodes a single `builds` edge from a fixed target → fixed image; the
  real per-target parsing lands in the dialect slice.

### C4 — GitHub Actions CI extractor (the publish leg)

- New `file_type = "ci"` (whitelisted) for workflow/job nodes; images stay `image` (C1).
- A workflow file (`detect_dialect` gains a CI branch, or a dedicated `extract_ci`) emits:
  - one node per job, `label = <job key or name>`, `file_type = "ci"`, `source_file = str(path)`;
  - a `publishes` edge `{relation: "publishes", confidence: "EXTRACTED"}` from the job node to each
    image the job's `docker build`/`docker push`/`build-and-push`/`docker/build-push-action` steps name
    concretely (registry path present, no `${{ vars }}`/`${{ secrets }}` unresolved);
  - a `references` edge from the job node to each `uses: actions/checkout@…` / `uses: ./.github/actions/…`
    target (the action reference), EXTRACTED, so workflow composition is navigable.
- Steps whose image is only `${{ vars.X }}` / `${{ secrets.X }}` are skipped (no placeholder image node)
  — mirror of I5.

### C5 — Cross-repo join (the uber-graph contract)

- Given two repos that reference the same image `registry_path`, running `graphify global add` for each
  produces a **single** image node in the global graph, with the `runs`/`publishes`/`builds` edges from
  *both* repos attached — a real cross-repo traversal.
- No extractor id contains a repo tag (I3); `global_add` remains the only place `repo_tag::` is added.
- The join is realized by the *existing* `global_add` external-node label-merge (`global_graph.py:123-134`):
  image nodes carry `source_file: None` and `label == registry_path`, so they dedup across repos.
  **No new global-graph plumbing is required for this contract.**

---

## 4. Test Targets (red-first)

Each target is one behavior; `[new]` = expected to fail first (behavior unimplemented);
`[guard]` = already-true property folded into a lock + one teeth-check. One contract = ≤4 targets.

**C1 — image node**
- T1 `[new]` — a concrete `image:` string with a `:tag` yields one image node with id/label = the
  tag-less registry path, `file_type="image"`, `source_file is None`, and `tags` containing the tag.
- T2 `[new]` — the same registry path seen once with `:tag` and once with `@sha256:…` produces **one**
  node id (identity is tag-less) but both tag strings collected in `attributes["tags"]`.
- T3 `[new]` — `file_type="image"` passes `validate_extraction` and survives `build_from_json`
  (adds to both whitelists; the existing k8s/argo/kustomize whitelist tests are the shape).

**C2 — k8s image capture**
- T4 `[new]` — a Pod/Deployment container `image: europe-west1-docker.pkg.dev/…/utility:tag` emits a
  `runs` edge from the workload node to `image://…/utility`, EXTRACTED, five fields.
- T5 `[new]` — a container `image: _IMAGE` (or `DEV_IMAGE` / `ko://…` / a bare name) emits **no**
  image node and **no** `runs` edge.

**C3 — Makefile/Dockerfile**
- T6 `[new]` — a `Makefile` with a target that references a concrete image emits a `builds` edge to
  that image node, `file_type="build"`, source_file = the Makefile path.
- T7 `[new]` — a `Dockerfile` `FROM <concrete image>` emits a `builds` edge to that image node.
- T8 `[new]` — a `.mk` (or extensionless `Makefile`) file is classified into the code path and
  produces ≥1 node (not "unclassified"); `file_type="build"` passes validate+build.

**C4 — GitHub Actions**
- T9 `[new]` — a workflow with a `docker build/push` step naming a concrete image emits a `publishes`
  edge from the job node to that image node, EXTRACTED.
- T10 `[new]` — a workflow `uses: actions/checkout@…` step emits a `references` edge from the job
  node to the action target; a `${{ vars.X }}` image is skipped (no placeholder).

**C5 — cross-repo join**
- T11 `[new]` — two synthetic repos (`builder`, `cloud`), each referencing the same image, run through
  `global add` → the global graph has **one** image node `repo_tag::…/schema` (deduped), and the
  `runs` edge (from cloud's Deployment) and `publishes` edge (from builder's CI) both point at it.
- T12 `[new]` — the resulting graph has a 2-hop path `Deployment -> Image -> CI-job` — a real
  cross-repo traversal (assert connectivity, not id text).

Budget: 12 targets across 5 contracts (C1:3, C2:2, C3:3, C4:2, C5:2). Coarser targets fold
multi-step behavior (T4 pins image-node *and* edge in one assertion; T11 pins dedup *and* both-edge
rewiring in one). T3/T8 are the whitelist guards folded into C1/C3 respectively.

---

## 5. Escalation Map (ordered task breakdown)

Each slice replaces one hardcoded stub with real behavior. Slices are sized as one coherent
capability; the downstream implementer decomposes each into 1–3 descending tests.

**Slice 1 — R1: the image node + whitelist (C1, T1-T3).**
Replace "no such node exists" with the image node shape. Green step hardcodes the C1 shape;
the real registry-path/tag parsing is exercised directly in T1/T2. Done when T1-T3 pass and the
full suite is green (whitelist add is the only build/validate change).

**Slice 2 — R2: k8s image capture (C2, T4-T5).**
Backfill `_extract_k8s` to capture each container `image:` (classified concrete-vs-placeholder) and emit
`runs` + image nodes. Done when T4-T5 pass with no k8s regression; `_container_names` still works.

**Slice 3 — R3: Makefile/.mk/Dockerfile extractor (C3, T6-T8).**
Classify `Makefile`/`Dockerfile`/`.mk` into code path; new `extract_makefile`/`_extract_build` emits
build nodes + `builds` edges. Done when T6-T8 pass; CI-unrelated `.mk`/`Makefile` still classified.
*(This is the genuinely-greenfield dialect; keep the first cut to targets that name images — no full Make grammar.)*

**Slice 4 — R4: GitHub Actions extractor (C5→C4, T9-T10).**
`detect_dialect` CI branch (or `extract_ci`) emits job nodes, `publishes` (docker build/push) and
`references` (uses) edges. Done when T9-T10 pass; non-CI `.yml` still a no-op (I1).

**Slice 5 — R5: the cross-repo join fixture (C5, T11-T12).**
No production code — a composition fixture: two synthetic repos through `global add`, asserting dedup +
cross-repo traversal. This is the slice that *proves the thread*. Done when T11-T12 pass against the
already-shipped `global_add`; any failure here is a C1/C2/C4 contract violation, not new work.

Order rationale: C1 (the join key) must land before any producer/consumer leg; C2 (consumer) before
C5 (the join needs a consumer edge); C3/C4 (producers) are independent of each other and of C2 once
C1 exists — they can go parallel (see §6). C5 last, as the proving composition.

---

## 6. Parallelization Matrix

**Safe to parallelize** (all locked against §3 contracts):
- **R3 (Makefile/.mk/Dockerfile)** ∥ **R4 (CI)** — both produce `builds`/`publishes` edges to the
  already-contracted C1 image node; neither touches the other's extractor or the k8s path.
- **R2 (k8s capture)** ∥ **R3/R4** once **R1** is merged — R2 is the consumer leg; R3/R4 are producer
  legs; all three only ever *reference* the C1 node shape. (R2 depends on R1 for the node; R3/R4 depend
  on R1 for the node.)

**Must stay serial:**
- **R1 before everything** — it defines the image node contract every other leg targets.
- **R5 after R2+R3+R4** — the join fixture needs all three legs to assert a real traversal; if any
  leg is missing, T11/T12 cannot pass. It moves no contract, but it *observes* all of them.

---

## 7. Invariants (hold across every slice)

- **I1 — Additive.** A repo with no images/CI/Makefiles is a no-op; existing k8s/argo/kustomize/terraform
  paths are byte-for-byte unchanged (no new edges on plain k8s beyond the added `runs` leg).
  *Guarded by the full-suite run after every slice.*
- **I2 — Determinism.** Byte-identical output across runs for every new extractor (image, build, ci);
  only ordered structures iterated.
- **I3 — No repo identity in extractor ids.** `image://`, `makefile://`, and CI node ids never contain a
  repo tag; `global_add` alone adds `repo_tag::`. *Guarded by T11 asserting the id has no repo text pre-merge.*
- **I4 — Image identity is the tag-less registry path.** Two refs to the same path with different
  tags/digests are one node; tags are attributes. *Guarded by T2.*
- **I5 — Placeholders never become image nodes.** `DEV_IMAGE`, `_IMAGE`, `_NEO4J_IMAGE_ID`, `ko://…`,
  bare names, and `${{ vars.* }}`/`${{ secrets.* }}` are skipped, never synthesized into a fake image.
  *Guarded by T5 and T10.*
- **I6 — Confidence integrity.** Every `runs`/`builds`/`publishes`/`references` edge resolves to
  EXTRACTED or is not emitted; nothing silently dropped to a dangling id (a placeholder, where needed,
  is a real node — but images are never placeholders; they are skipped per I5). *Guarded by T4/T9/T10.*
- **I7 — `file_type`s never degrade.** `image`, `build`, `ci` survive validate+build and push as
  `Image`/`Build`/`Ci` (existing `.capitalize()` path; `ci`→`Ci` is accepted). *Guarded by T3/T8, and
  the existing `K8s`/`Argo`/`Kustomize` label-pin pattern.*

---

## 8. Requirement backlog

- [ ] R1 — Image node + `file_type="image"` whitelist (C1): id/label = tag-less registry path,
  `source_file None`, tags in attributes. `depends: none; track: serial`
- [ ] R2 — k8s `image:` capture (C2): `runs` edges from workload→image, concrete-vs-placeholder
  classification. `depends: R1; track: serial`
- [ ] R3 — Makefile/`.mk`/Dockerfile extractor (C3): code-path classification + `builds` edges,
  `file_type="build"`. `depends: R1; track: parallel-with R4`
- [ ] R4 — GitHub Actions CI extractor (C4): job nodes, `publishes` + `references` edges,
  `file_type="ci"`. `depends: R1; track: parallel-with R3`
- [ ] R5 — Cross-repo join fixture (C5): two synthetic repos through `global add` → one deduped
  image node with real cross-repo `runs`/`publishes` traversal. `depends: R2, R3, R4; track: serial`
- [ ] R6 — (deferred, §2) cross-repo *component/library* link — `kg-builder`→`neo4j-graphrag` code
  identity. `depends: R5; track: serial — after the thread`
- [ ] R7 — (deferred, §2) GitLab CI + docker-compose + Ansible dialects, reusing C1/C4 image-join.
  `depends: R5; track: parallel-with R6`

**Bootstrap note for the orchestrator.** §3 (Contracts) + §7 (Invariants) are the always-loaded
cross-cutting file. §8 is the one-at-a-time backlog. An implementer on R1 reads §3-C1 + §7-I3/I4/I7
— not R2-R7.
