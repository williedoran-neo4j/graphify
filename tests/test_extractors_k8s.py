"""Tests for the Kubernetes YAML extractor (graphify/extractors/k8s.py)."""
from __future__ import annotations

from pathlib import Path

from graphify.extractors.k8s import (
    Dialect,
    _resolve_argo_references,
    _resolve_k8s_references,
    _resolve_kustomize_includes,
    detect_dialect,
    extract_k8s,
)


def test_detect_dialect_classifies_k8s_manifests_and_rejects_everything_else():
    """detect_dialect returns K8S_MANIFEST only when every YAML document is a
    mapping with string-valued apiVersion and kind, and no apiVersion starts with
    argoproj.io/. All other inputs return None."""
    p = Path("/dev/null")

    # Empty / whitespace must return None (not vacuously True).
    assert detect_dialect(p, "") is None
    assert detect_dialect(p, "   \n\n   ") is None

    # Valid single-document manifest.
    single = """\
apiVersion: v1
kind: Pod
metadata:
  name: nginx
"""
    assert detect_dialect(p, single) is Dialect.K8S_MANIFEST

    # Valid multi-document manifest.
    multi = """\
apiVersion: v1
kind: Pod
metadata:
  name: a
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: b
"""
    assert detect_dialect(p, multi) is Dialect.K8S_MANIFEST

    # Missing apiVersion.
    missing_api = """\
kind: Pod
metadata:
  name: nginx
"""
    assert detect_dialect(p, missing_api) is None

    # Missing kind.
    missing_kind = """\
apiVersion: v1
metadata:
  name: nginx
"""
    assert detect_dialect(p, missing_kind) is None

    # Non-string apiVersion.
    bad_api = """\
apiVersion: 123
kind: Pod
"""
    assert detect_dialect(p, bad_api) is None

    # Non-string kind.
    bad_kind = """\
apiVersion: v1
kind: 456
"""
    assert detect_dialect(p, bad_kind) is None

    # Argo CD manifest (apiVersion starts with argoproj.io/).
    argo = """\
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
"""
    assert detect_dialect(p, argo) is None

    # Unparseable YAML.
    assert detect_dialect(p, "bad: { broken: yaml: here") is None

    # Top-level document is not a mapping.
    list_doc = """\
- just
- a
- list
"""
    assert detect_dialect(p, list_doc) is None

    # Mixed multi-doc: one valid, one invalid.
    mixed = """\
apiVersion: v1
kind: Pod
metadata:
  name: a
---
kind: MissingApiVersion
"""
    assert detect_dialect(p, mixed) is None


def test_detect_dialect_classifies_kustomization_and_rejects_mixed_and_non_kustomization_kinds():
    """detect_dialect returns KUSTOMIZATION only when every YAML document has an
    apiVersion starting with kustomize.config.k8s.io/ and kind == Kustomization.
    Mixed files, plain K8s manifests, and Argo manifests return None or their
    respective dialects."""
    p = Path("/dev/null")

    # Single Kustomization — valid kustomization dialect.
    kustomization = """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../base
"""
    assert detect_dialect(p, kustomization) is Dialect.KUSTOMIZATION

    # Multi-document Kustomization — still valid when every doc is Kustomization.
    multi_kustomization = """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ./api
---
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ./web
"""
    assert detect_dialect(p, multi_kustomization) is Dialect.KUSTOMIZATION

    # Mixed multi-doc: Kustomization + plain K8s ConfigMap → None.
    mixed_kustomization_k8s = """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../base
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
"""
    assert detect_dialect(p, mixed_kustomization_k8s) is None

    # Argo workflow with kustomize-like name but argo apiVersion → ARGO_WORKFLOW.
    argo_workflow = """\
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: my-flow
"""
    assert detect_dialect(p, argo_workflow) is Dialect.ARGO_WORKFLOW

    # Plain K8s still covered by other tests; not re-asserted here.


def test_detect_dialect_classifies_argo_workflow_manifests_and_rejects_mixed_and_non_workflow_kinds():
    """detect_dialect returns ARGO_WORKFLOW only when every YAML document has an
    apiVersion starting with argoproj.io/ and a kind in the workflow set.
    Mixed files, non-workflow Argo kinds, and plain K8s manifests return None."""
    p = Path("/dev/null")

    # Single WorkflowTemplate — valid Argo workflow.
    workflow_template = """\
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: my-template
"""
    assert detect_dialect(p, workflow_template) is Dialect.ARGO_WORKFLOW

    # Single CronWorkflow — valid Argo workflow.
    cron_workflow = """\
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: my-cron
"""
    assert detect_dialect(p, cron_workflow) is Dialect.ARGO_WORKFLOW

    # Mixed multi-doc: one Argo Workflow + one plain K8s ConfigMap.
    mixed_argo_k8s = """\
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: my-flow
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
"""
    assert detect_dialect(p, mixed_argo_k8s) is None

    # Argo CD Application — NOT a workflow kind, must stay None.
    argo_app = """\
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
"""
    assert detect_dialect(p, argo_app) is None


def test_detect_dialect_classifies_ci_workflow_and_rejects_mixed_and_k8s_with_jobs():
    """detect_dialect returns CI_WORKFLOW for a GitHub Actions YAML with a top-level
    dict containing a 'jobs' key and no apiVersion or kind. Mixed documents
    (workflow + k8s manifest) return None, and a k8s manifest that also contains a
    'jobs' key but has apiVersion/kind is classified as K8S_MANIFEST, not CI."""
    p = Path("/dev/null")

    # Single GitHub Actions workflow — valid CI dialect.
    ci_workflow = """\
name: CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""
    assert detect_dialect(p, ci_workflow) is Dialect.CI_WORKFLOW

    # Multi-document: workflow + k8s manifest → mixed, so None.
    mixed_ci_k8s = """\
name: CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
---
apiVersion: v1
kind: Pod
metadata:
  name: nginx
"""
    assert detect_dialect(p, mixed_ci_k8s) is None

    # A k8s manifest with a 'jobs' key but also apiVersion/kind → K8S_MANIFEST, not CI.
    k8s_with_jobs = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: my-job
spec:
  template:
    spec:
      containers:
        - name: app
"""
    assert detect_dialect(p, k8s_with_jobs) is Dialect.K8S_MANIFEST


def test_extract_k8s_emits_raw_id_nodes_with_attributes_and_ignores_non_k8s():
    """extract_k8s emits one node per k8s manifest doc with raw k8s:// ids,
    labels, file_type, source metadata, and attributes including containers.
    Unknown kinds and namespace-less resources are accepted. Non-k8s YAML
    returns empty nodes and edges."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 1. Namespaced Deployment with mixed-case name and hyphen.
        deployment = td_path / "deployment.yaml"
        deployment.write_text(
            """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: payments
spec:
  template:
    spec:
      containers:
        - name: app
        - name: sidecar
""",
            encoding="utf-8",
        )

        # 2. Cluster-scoped / namespace-less resource (no metadata.namespace).
        cluster_role = td_path / "clusterrole.yaml"
        cluster_role.write_text(
            """\
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: reader
""",
            encoding="utf-8",
        )

        # 3. Unknown/CRD kind (no containers, but still emitted).
        custom = td_path / "custom.yaml"
        custom.write_text(
            """\
apiVersion: example.com/v1
kind: MyCustomResource
metadata:
  name: my-obj
  namespace: custom-ns
""",
            encoding="utf-8",
        )

        # 4. Non-k8s YAML (GitHub Actions workflow, no apiVersion/kind).
        non_k8s = td_path / "workflow.yaml"
        non_k8s.write_text(
            """\
name: CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
""",
            encoding="utf-8",
        )

        result_dep = extract_k8s(deployment)
        result_cr = extract_k8s(cluster_role)
        result_custom = extract_k8s(custom)
        result_non = extract_k8s(non_k8s)

        # Workflow with one job now yields one ci node (not empty after C3).
        assert len(result_non["nodes"]) == 1
        ci_node = result_non["nodes"][0]
        assert ci_node["file_type"] == "ci"
        assert ci_node["label"] == "build"
        assert ci_node["source_file"] == str(non_k8s)
        assert ci_node["source_location"] == "doc0"
        assert ci_node["id"].startswith("ci://")
        assert result_non["edges"] == []

        # Deployment assertions.
        dep_nodes = result_dep["nodes"]
        assert len(dep_nodes) == 1
        dep = dep_nodes[0]
        assert dep["id"] == "k8s://payments/Deployment/api-server"
        assert dep["label"] == "Deployment/api-server"
        assert dep["file_type"] == "k8s"
        assert dep["source_file"] == str(deployment)
        assert dep["source_location"] == "doc0"
        assert dep["attributes"] == {
            "kind": "Deployment",
            "namespace": "payments",
            "containers": ["app", "sidecar"],
        }
        assert result_dep["edges"] == []

        # ClusterRole assertions (no namespace → "_cluster", no containers).
        cr_nodes = result_cr["nodes"]
        assert len(cr_nodes) == 1
        cr = cr_nodes[0]
        assert cr["id"] == "k8s://_cluster/ClusterRole/reader"
        assert cr["label"] == "ClusterRole/reader"
        assert cr["file_type"] == "k8s"
        assert cr["source_file"] == str(cluster_role)
        assert cr["source_location"] == "doc0"
        assert cr["attributes"] == {
            "kind": "ClusterRole",
            "namespace": "_cluster",
        }
        assert "containers" not in cr["attributes"]
        assert result_cr["edges"] == []

        # Custom resource assertions.
        custom_nodes = result_custom["nodes"]
        assert len(custom_nodes) == 1
        cu = custom_nodes[0]
        assert cu["id"] == "k8s://custom-ns/MyCustomResource/my-obj"
        assert cu["label"] == "MyCustomResource/my-obj"
        assert cu["file_type"] == "k8s"
        assert cu["source_file"] == str(custom)
        assert cu["source_location"] == "doc0"
        assert cu["attributes"] == {
            "kind": "MyCustomResource",
            "namespace": "custom-ns",
        }
        assert "containers" not in cu["attributes"]
        assert result_custom["edges"] == []


def test_k8s_file_type_passes_validation_and_survives_build():
    """A node with file_type="k8s" must pass schema validation without a file_type
    error and must retain file_type="k8s" through graph assembly, not be coerced
    to "concept"."""
    from graphify.validate import validate_extraction
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "k8s://x/K/y",
                "label": "K/y",
                "file_type": "k8s",
                "source_file": "f.yaml",
            }
        ],
        "edges": [],
    }

    # Validation: must NOT flag "k8s" as an invalid file_type
    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'k8s'" in e]
    assert file_type_errors == []

    # Build: file_type must survive as "k8s", not be coerced to "concept"
    G = build_from_json(extraction)
    assert G.nodes["k8s://x/K/y"]["file_type"] == "k8s"


def test_argo_file_type_passes_validation_and_survives_build():
    """A node with file_type="argo" must pass schema validation without a file_type
    error and must retain file_type="argo" through graph assembly, not be coerced
    to "concept"."""
    from graphify.validate import validate_extraction
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "argo://x/WorkflowTemplate/n",
                "label": "WorkflowTemplate/n",
                "file_type": "argo",
                "source_file": "f.yaml",
            }
        ],
        "edges": [],
    }

    # Validation: must NOT flag "argo" as an invalid file_type
    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'argo'" in e]
    assert file_type_errors == []

    # Build: file_type must survive as "argo", not be coerced to "concept"
    G = build_from_json(extraction)
    assert G.nodes["argo://x/WorkflowTemplate/n"]["file_type"] == "argo"


def test_kustomize_file_type_survives_build():
    """A node with file_type="kustomize" must retain file_type="kustomize"
    through graph assembly, not be coerced to "concept"."""
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "kustomize://base/kustomization",
                "label": "base",
                "file_type": "kustomize",
                "source_file": "kustomization.yaml",
            }
        ],
        "edges": [],
    }

    G = build_from_json(extraction)
    assert G.nodes["kustomize://base/kustomization"]["file_type"] == "kustomize"


def test_extract_k8s_collects_candidates_for_references_and_service_account():
    """extract_k8s returns a k8s_candidates list with one bare dict per
    reference (envFrom, env.valueFrom, volumes) and per serviceAccountName.
    Each candidate carries target_name, target_kind, namespace, relation,
    source_file, source_kind, and source_name — no edge-shaped keys."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Deployment with every reference source and a serviceAccountName.
        deployment = td_path / "deployment.yaml"
        deployment.write_text(
            """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: payments
spec:
  template:
    spec:
      serviceAccountName: my-sa
      containers:
        - name: app
          envFrom:
            - configMapRef:
                name: envfrom-cm
            - secretRef:
                name: envfrom-sec
          env:
            - name: KEY1
              valueFrom:
                configMapKeyRef:
                  name: env-cm-key
                  key: k1
            - name: KEY2
              valueFrom:
                secretKeyRef:
                  name: env-sec-key
                  key: k2
          volumeMounts:
            - name: vol-cm
              mountPath: /etc/cm
            - name: vol-sec
              mountPath: /etc/secret
      volumes:
        - name: vol-cm
          configMap:
            name: vol-cm-name
        - name: vol-sec
          secret:
            secretName: vol-sec-name
""",
            encoding="utf-8",
        )

        # Pod with direct serviceAccountName (not under template).
        pod = td_path / "pod.yaml"
        pod.write_text(
            """\
apiVersion: v1
kind: Pod
metadata:
  name: worker
  namespace: jobs
spec:
  serviceAccountName: pod-sa
  containers:
    - name: main
""",
            encoding="utf-8",
        )

        # K8s manifest with no references at all.
        bare = td_path / "bare.yaml"
        bare.write_text(
            """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: plain-cm
  namespace: default
""",
            encoding="utf-8",
        )

        result_dep = extract_k8s(deployment)
        result_pod = extract_k8s(pod)
        result_bare = extract_k8s(bare)

        # edges must stay empty (candidates are not edges yet).
        assert result_dep["edges"] == []
        assert result_pod["edges"] == []
        assert result_bare["edges"] == []

        # --- Deployment candidates ---
        cands = result_dep["k8s_candidates"]
        assert len(cands) == 7

        expected_dep = [
            {
                "target_name": "envfrom-cm",
                "target_kind": "ConfigMap",
                "namespace": "payments",
                "relation": "references",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
            {
                "target_name": "envfrom-sec",
                "target_kind": "Secret",
                "namespace": "payments",
                "relation": "references",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
            {
                "target_name": "env-cm-key",
                "target_kind": "ConfigMap",
                "namespace": "payments",
                "relation": "references",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
            {
                "target_name": "env-sec-key",
                "target_kind": "Secret",
                "namespace": "payments",
                "relation": "references",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
            {
                "target_name": "vol-cm-name",
                "target_kind": "ConfigMap",
                "namespace": "payments",
                "relation": "references",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
            {
                "target_name": "vol-sec-name",
                "target_kind": "Secret",
                "namespace": "payments",
                "relation": "references",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
            {
                "target_name": "my-sa",
                "target_kind": "ServiceAccount",
                "namespace": "payments",
                "relation": "uses_service_account",
                "source_file": str(deployment),
                "source_kind": "Deployment",
                "source_name": "api-server",
            },
        ]
        for exp in expected_dep:
            assert any(c == exp for c in cands), f"Missing candidate {exp!r}"

        # Every candidate must have exactly the allowed keys (no edge-shaped keys).
        for c in cands:
            assert set(c.keys()) == {
                "target_name",
                "target_kind",
                "namespace",
                "relation",
                "source_file",
                "source_kind",
                "source_name",
            }

        # --- Pod candidates ---
        pod_cands = result_pod["k8s_candidates"]
        assert len(pod_cands) == 1
        assert pod_cands[0] == {
            "target_name": "pod-sa",
            "target_kind": "ServiceAccount",
            "namespace": "jobs",
            "relation": "uses_service_account",
            "source_file": str(pod),
            "source_kind": "Pod",
            "source_name": "worker",
        }

        # --- Bare manifest (no references) ---
        assert result_bare["k8s_candidates"] == []


def test_yaml_is_code_and_dispatched_to_extract_k8s():
    """.yaml and .yml must be classified as code (not documents) and routed to
    extract_k8s by _get_extractor."""
    from graphify.detect import CODE_EXTENSIONS, DOC_EXTENSIONS
    from graphify.extract import _get_extractor

    # Reclassification boundary: YAML is code, not a document.
    assert ".yaml" in CODE_EXTENSIONS
    assert ".yml" in CODE_EXTENSIONS
    assert ".yaml" not in DOC_EXTENSIONS
    assert ".yml" not in DOC_EXTENSIONS

    # Dispatch boundary: _get_extractor must route to extract_k8s.
    assert _get_extractor(Path("deploy.yaml")) is extract_k8s
    assert _get_extractor(Path("setup.yml")) is extract_k8s


def test_resolve_k8s_references_emits_exact_references_edges_with_extracted_confidence():
    """Pass 2 resolution: candidates for same-namespace references become edges
    with all five required fields and EXTRACTED confidence. Non-k8s nodes are
    ignored by the index. No candidates means no edges appended."""
    all_nodes = [
        {
            "id": "k8s://payments/Deployment/api-server",
            "label": "Deployment/api-server",
            "file_type": "k8s",
            "source_file": "dep.yaml",
            "attributes": {"kind": "Deployment", "namespace": "payments"},
        },
        {
            "id": "k8s://payments/ConfigMap/envfrom-cm",
            "label": "ConfigMap/envfrom-cm",
            "file_type": "k8s",
            "source_file": "cm.yaml",
            "attributes": {"kind": "ConfigMap", "namespace": "payments"},
        },
        {
            "id": "k8s://payments/Secret/envfrom-sec",
            "label": "Secret/envfrom-sec",
            "file_type": "k8s",
            "source_file": "sec.yaml",
            "attributes": {"kind": "Secret", "namespace": "payments"},
        },
        # Non-k8s node must not pollute the index.
        {
            "id": "python://payments/whatever.py::foo",
            "label": "foo",
            "file_type": "python",
            "source_file": "whatever.py",
        },
    ]

    per_file = [
        {
            "nodes": [],
            "edges": [],
            "k8s_candidates": [
                {
                    "target_name": "envfrom-cm",
                    "target_kind": "ConfigMap",
                    "namespace": "payments",
                    "relation": "references",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "api-server",
                },
                {
                    "target_name": "envfrom-sec",
                    "target_kind": "Secret",
                    "namespace": "payments",
                    "relation": "references",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "api-server",
                },
            ],
        }
    ]

    all_edges: list[dict] = []
    _resolve_k8s_references(per_file, all_nodes, all_edges)

    assert len(all_edges) == 2

    cm_edge = all_edges[0]
    assert cm_edge == {
        "source": "k8s://payments/Deployment/api-server",
        "target": "k8s://payments/ConfigMap/envfrom-cm",
        "relation": "references",
        "confidence": "EXTRACTED",
        "source_file": "dep.yaml",
    }

    sec_edge = all_edges[1]
    assert sec_edge == {
        "source": "k8s://payments/Deployment/api-server",
        "target": "k8s://payments/Secret/envfrom-sec",
        "relation": "references",
        "confidence": "EXTRACTED",
        "source_file": "dep.yaml",
    }

    # --- No candidates: nothing appended ---
    empty_per_file = [{"nodes": [], "edges": [], "k8s_candidates": []}]
    empty_edges: list[dict] = []
    _resolve_k8s_references(empty_per_file, all_nodes, empty_edges)
    assert empty_edges == []


def test_resolve_k8s_references_emits_ambiguous_edges_and_placeholder_nodes_for_unresolved():
    """Unresolved references (absent or in a different namespace) emit an AMBIGUOUS
    edge with a #unresolved target id, plus a deduplicated placeholder node for
    that target. Resolved references still produce EXTRACTED edges and no
    placeholder."""
    all_nodes = [
        # Resolved: same-namespace target.
        {
            "id": "k8s://payments/ConfigMap/same-ns-cm",
            "label": "ConfigMap/same-ns-cm",
            "file_type": "k8s",
            "source_file": "cm.yaml",
            "attributes": {"kind": "ConfigMap", "namespace": "payments"},
        },
        # Wrong-namespace target: exists in "other", not in "payments".
        {
            "id": "k8s://other/ConfigMap/envfrom-cm",
            "label": "ConfigMap/envfrom-cm",
            "file_type": "k8s",
            "source_file": "other/cm.yaml",
            "attributes": {"kind": "ConfigMap", "namespace": "other"},
        },
    ]

    per_file = [
        {
            "nodes": [],
            "edges": [],
            "k8s_candidates": [
                # 1. Resolved cleanly → EXTRACTED edge, no placeholder.
                {
                    "target_name": "same-ns-cm",
                    "target_kind": "ConfigMap",
                    "namespace": "payments",
                    "relation": "references",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "api-server",
                },
                # 2. Wrong-namespace → AMBIGUOUS edge + placeholder.
                {
                    "target_name": "envfrom-cm",
                    "target_kind": "ConfigMap",
                    "namespace": "payments",
                    "relation": "references",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "api-server",
                },
                # 3. Absent entirely → AMBIGUOUS edge + placeholder.
                {
                    "target_name": "missing-cm",
                    "target_kind": "ConfigMap",
                    "namespace": "payments",
                    "relation": "references",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "api-server",
                },
                # 4. Duplicate absent reference → AMBIGUOUS edge, but placeholder
                #    must be deduplicated with #3.
                {
                    "target_name": "missing-cm",
                    "target_kind": "ConfigMap",
                    "namespace": "payments",
                    "relation": "references",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "sidecar",
                },
            ],
        }
    ]

    all_edges: list[dict] = []
    _resolve_k8s_references(per_file, all_nodes, all_edges)

    # --- Edges: 1 EXTRACTED + 3 AMBIGUOUS ---
    assert len(all_edges) == 4

    extracted = [e for e in all_edges if e.get("confidence") == "EXTRACTED"]
    ambiguous = [e for e in all_edges if e.get("confidence") == "AMBIGUOUS"]
    assert len(extracted) == 1
    assert len(ambiguous) == 3

    # EXTRACTED edge for same-ns-cm (target is the real node id).
    assert extracted[0] == {
        "source": "k8s://payments/Deployment/api-server",
        "target": "k8s://payments/ConfigMap/same-ns-cm",
        "relation": "references",
        "confidence": "EXTRACTED",
        "source_file": "dep.yaml",
    }

    # AMBIGUOUS edges: two for missing-cm (deduped placeholder), one for envfrom-cm.
    wrong_ns_edge = next(
        e for e in ambiguous
        if e["target"] == "k8s://payments/ConfigMap/envfrom-cm#unresolved"
    )
    assert wrong_ns_edge == {
        "source": "k8s://payments/Deployment/api-server",
        "target": "k8s://payments/ConfigMap/envfrom-cm#unresolved",
        "relation": "references",
        "confidence": "AMBIGUOUS",
        "source_file": "dep.yaml",
    }

    missing_edges = [e for e in ambiguous if e["target"] == "k8s://payments/ConfigMap/missing-cm#unresolved"]
    assert len(missing_edges) == 2
    assert missing_edges[0] == {
        "source": "k8s://payments/Deployment/api-server",
        "target": "k8s://payments/ConfigMap/missing-cm#unresolved",
        "relation": "references",
        "confidence": "AMBIGUOUS",
        "source_file": "dep.yaml",
    }
    assert missing_edges[1] == {
        "source": "k8s://payments/Deployment/sidecar",
        "target": "k8s://payments/ConfigMap/missing-cm#unresolved",
        "relation": "references",
        "confidence": "AMBIGUOUS",
        "source_file": "dep.yaml",
    }

    # --- Placeholder nodes ---
    placeholders = [n for n in all_nodes if n.get("id", "").endswith("#unresolved")]
    assert len(placeholders) == 2, "Only two unique unresolved targets, so two placeholders"

    envfrom_ph = next(
        n for n in placeholders
        if n["id"] == "k8s://payments/ConfigMap/envfrom-cm#unresolved"
    )
    assert envfrom_ph == {
        "id": "k8s://payments/ConfigMap/envfrom-cm#unresolved",
        "label": "ConfigMap/envfrom-cm (unresolved)",
        "file_type": "k8s",
        "source_file": "dep.yaml",
        "attributes": {"unresolved": True},
    }

    missing_ph = next(
        n for n in placeholders
        if n["id"] == "k8s://payments/ConfigMap/missing-cm#unresolved"
    )
    assert missing_ph == {
        "id": "k8s://payments/ConfigMap/missing-cm#unresolved",
        "label": "ConfigMap/missing-cm (unresolved)",
        "file_type": "k8s",
        "source_file": "dep.yaml",
        "attributes": {"unresolved": True},
    }


def test_resolve_k8s_references_emits_uses_service_account_edges_extracted_and_ambiguous():
    """uses_service_account candidates resolve through the same EXTRACTED/AMBIGUOUS
    path as references: a same-namespace ServiceAccount yields an EXTRACTED edge,
    an absent one yields an AMBIGUOUS edge plus a deduped placeholder node, and no
    candidate is silently dropped."""
    all_nodes = [
        {
            "id": "k8s://payments/Deployment/api-server",
            "label": "Deployment/api-server",
            "file_type": "k8s",
            "source_file": "dep.yaml",
            "attributes": {"kind": "Deployment", "namespace": "payments"},
        },
        {
            "id": "k8s://payments/ServiceAccount/my-sa",
            "label": "ServiceAccount/my-sa",
            "file_type": "k8s",
            "source_file": "sa.yaml",
            "attributes": {"kind": "ServiceAccount", "namespace": "payments"},
        },
        {
            "id": "k8s://jobs/Pod/worker",
            "label": "Pod/worker",
            "file_type": "k8s",
            "source_file": "pod.yaml",
            "attributes": {"kind": "Pod", "namespace": "jobs"},
        },
    ]

    per_file = [
        {
            "nodes": [],
            "edges": [],
            "k8s_candidates": [
                # EXTRACTED — same-namespace ServiceAccount exists.
                {
                    "target_name": "my-sa",
                    "target_kind": "ServiceAccount",
                    "namespace": "payments",
                    "relation": "uses_service_account",
                    "source_file": "dep.yaml",
                    "source_kind": "Deployment",
                    "source_name": "api-server",
                },
                # AMBIGUOUS — ServiceAccount absent in jobs namespace.
                {
                    "target_name": "pod-sa",
                    "target_kind": "ServiceAccount",
                    "namespace": "jobs",
                    "relation": "uses_service_account",
                    "source_file": "pod.yaml",
                    "source_kind": "Pod",
                    "source_name": "worker",
                },
            ],
        }
    ]

    all_edges: list[dict] = []
    _resolve_k8s_references(per_file, all_nodes, all_edges)

    # Exactly one edge per candidate — no silent drop.
    assert len(all_edges) == 2

    extracted = [e for e in all_edges if e.get("confidence") == "EXTRACTED"]
    ambiguous = [e for e in all_edges if e.get("confidence") == "AMBIGUOUS"]
    assert len(extracted) == 1
    assert len(ambiguous) == 1

    assert extracted[0] == {
        "source": "k8s://payments/Deployment/api-server",
        "target": "k8s://payments/ServiceAccount/my-sa",
        "relation": "uses_service_account",
        "confidence": "EXTRACTED",
        "source_file": "dep.yaml",
    }

    assert ambiguous[0] == {
        "source": "k8s://jobs/Pod/worker",
        "target": "k8s://jobs/ServiceAccount/pod-sa#unresolved",
        "relation": "uses_service_account",
        "confidence": "AMBIGUOUS",
        "source_file": "pod.yaml",
    }

    # Placeholder node for the unresolved ServiceAccount.
    placeholders = [n for n in all_nodes if n.get("id", "").endswith("#unresolved")]
    assert len(placeholders) == 1
    assert placeholders[0] == {
        "id": "k8s://jobs/ServiceAccount/pod-sa#unresolved",
        "label": "ServiceAccount/pod-sa (unresolved)",
        "file_type": "k8s",
        "source_file": "pod.yaml",
        "attributes": {"unresolved": True},
    }


def test_extract_k8s_captures_service_selector_and_workload_labels():
    """extract_k8s adds selector to Service nodes from spec.selector and labels
    to workload nodes from spec.template.metadata.labels (Deployment/StatefulSet)
    or bare metadata.labels (Pod). No selector/labels keys are added when absent,
    and a Service's own metadata.labels is never treated as a selector."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 1. Service with spec.selector (and metadata.labels to ensure it is ignored).
        svc = td_path / "service.yaml"
        svc.write_text(
            """\
apiVersion: v1
kind: Service
metadata:
  name: my-svc
  namespace: default
  labels:
    env: prod
spec:
  selector:
    app: web
    tier: frontend
""",
            encoding="utf-8",
        )

        # 2. Deployment with spec.template.metadata.labels.
        dep = td_path / "deployment.yaml"
        dep.write_text(
            """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-dep
  namespace: default
spec:
  template:
    metadata:
      labels:
        app: web
        version: v1
    spec:
      containers:
        - name: app
""",
            encoding="utf-8",
        )

        # 3. Pod with bare metadata.labels (no template).
        pod = td_path / "pod.yaml"
        pod.write_text(
            """\
apiVersion: v1
kind: Pod
metadata:
  name: web-pod
  namespace: default
  labels:
    app: web
    job: worker
spec:
  containers:
    - name: app
""",
            encoding="utf-8",
        )

        # 4. ConfigMap with no labels or selector.
        cm = td_path / "configmap.yaml"
        cm.write_text(
            """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
  namespace: default
""",
            encoding="utf-8",
        )

        # 5. Service with metadata.labels but no spec.selector.
        svc_no_selector = td_path / "service_no_selector.yaml"
        svc_no_selector.write_text(
            """\
apiVersion: v1
kind: Service
metadata:
  name: bare-svc
  namespace: default
  labels:
    env: prod
""",
            encoding="utf-8",
        )

        result_svc = extract_k8s(svc)
        result_dep = extract_k8s(dep)
        result_pod = extract_k8s(pod)
        result_cm = extract_k8s(cm)
        result_svc_no = extract_k8s(svc_no_selector)

        # --- Service with selector ---
        svc_nodes = result_svc["nodes"]
        assert len(svc_nodes) == 1
        svc_node = svc_nodes[0]
        assert svc_node["attributes"]["kind"] == "Service"
        assert svc_node["attributes"]["selector"] == {"app": "web", "tier": "frontend"}
        assert "labels" not in svc_node["attributes"]

        # --- Deployment with labels ---
        dep_nodes = result_dep["nodes"]
        assert len(dep_nodes) == 1
        dep_node = dep_nodes[0]
        assert dep_node["attributes"]["labels"] == {"app": "web", "version": "v1"}

        # --- Pod with labels ---
        pod_nodes = result_pod["nodes"]
        assert len(pod_nodes) == 1
        pod_node = pod_nodes[0]
        assert pod_node["attributes"]["labels"] == {"app": "web", "job": "worker"}

        # --- ConfigMap: no selector/labels added ---
        cm_nodes = result_cm["nodes"]
        assert len(cm_nodes) == 1
        cm_node = cm_nodes[0]
        assert "selector" not in cm_node["attributes"]
        assert "labels" not in cm_node["attributes"]

        # --- Service with metadata.labels but no spec.selector: no selector added ---
        svc_no_nodes = result_svc_no["nodes"]
        assert len(svc_no_nodes) == 1
        svc_no_node = svc_no_nodes[0]
        assert "selector" not in svc_no_node["attributes"]
        assert "labels" not in svc_no_node["attributes"]


def test_resolve_k8s_references_emits_selects_edge_for_subset_selector_match():
    """Services whose selector is a subset of a workload's labels in the same
    namespace emit a single selects edge with INFERRED confidence. No edge is
    emitted when the selector is not a subset or the namespaces differ."""
    svc_source_file = "svc.yaml"
    all_nodes = [
        {
            "id": "k8s://payments/Service/svc",
            "label": "Service/svc",
            "file_type": "k8s",
            "source_file": svc_source_file,
            "attributes": {
                "kind": "Service",
                "namespace": "payments",
                "selector": {"app": "api", "tier": "frontend"},
            },
        },
        {
            "id": "k8s://payments/Deployment/api",
            "label": "Deployment/api",
            "file_type": "k8s",
            "source_file": "dep.yaml",
            "attributes": {
                "kind": "Deployment",
                "namespace": "payments",
                "labels": {"app": "api", "tier": "frontend", "env": "prod"},
            },
        },
        {
            "id": "k8s://payments/Deployment/other",
            "label": "Deployment/other",
            "file_type": "k8s",
            "source_file": "other.yaml",
            "attributes": {
                "kind": "Deployment",
                "namespace": "payments",
                "labels": {"app": "other", "tier": "frontend"},
            },
        },
        {
            "id": "k8s://other/Deployment/api",
            "label": "Deployment/api",
            "file_type": "k8s",
            "source_file": "other_dep.yaml",
            "attributes": {
                "kind": "Deployment",
                "namespace": "other",
                "labels": {"app": "api", "tier": "frontend", "env": "prod"},
            },
        },
    ]

    per_file = []
    all_edges: list[dict] = []
    _resolve_k8s_references(per_file, all_nodes, all_edges)

    selects_edges = [e for e in all_edges if e.get("relation") == "selects"]
    assert len(selects_edges) == 1, f"Expected exactly 1 selects edge, got {len(selects_edges)}"

    edge = selects_edges[0]
    assert edge["source"] == "k8s://payments/Service/svc"
    assert edge["target"] == "k8s://payments/Deployment/api"
    assert edge["relation"] == "selects"
    assert edge["confidence"] == "INFERRED"
    assert edge["source_file"] == svc_source_file

    # Ensure no edges were emitted to the non-matching or cross-namespace workloads.
    targets = {e["target"] for e in all_edges}
    assert "k8s://payments/Deployment/other" not in targets
    assert "k8s://other/Deployment/api" not in targets


def test_extract_k8s_emits_argo_workflow_and_template_nodes_for_workflow_template():
    """An Argo WorkflowTemplate with entrypoint and templates emits a workflow-level
    node with argo:// id and file_type='argo', plus one template node per
    spec.templates[] entry with container_image only when a container or script
    image is present."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wftmpl = td_path / "hello-artifacts-wftmpl.yaml"
        wftmpl.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: hello-artifacts
  namespace: kg-builder
spec:
  entrypoint: main
  templates:
    - name: main
      dag:
        tasks:
          - name: produce
            template: produce
    - name: produce
      container:
        image: alpine:3.20
""",
            encoding="utf-8",
        )

        result = extract_k8s(wftmpl)


        nodes = result["nodes"]
        assert len(nodes) == 3, f"Expected 3 nodes (1 workflow + 2 templates), got {len(nodes)}"

        # Workflow-level node
        workflow_node = next(
            (n for n in nodes if n["id"] == "argo://kg-builder/WorkflowTemplate/hello-artifacts"),
            None,
        )
        assert workflow_node is not None
        assert workflow_node["label"] == "WorkflowTemplate/hello-artifacts"
        assert workflow_node["file_type"] == "argo"
        assert workflow_node["source_file"] == str(wftmpl)
        assert workflow_node["source_location"] == "doc0"
        assert workflow_node["attributes"]["kind"] == "WorkflowTemplate"
        assert workflow_node["attributes"]["namespace"] == "kg-builder"
        assert workflow_node["attributes"]["entrypoint"] == "main"

        # Template node: main (dag template, no container_image)
        main_template = next(
            (n for n in nodes if n["id"] == "argo://kg-builder/WorkflowTemplate/hello-artifacts/main"),
            None,
        )
        assert main_template is not None
        assert main_template["label"] == "main"
        assert main_template["file_type"] == "argo"
        assert main_template["source_file"] == str(wftmpl)
        assert main_template["source_location"] == "doc0"
        assert main_template["attributes"]["template"] == "main"
        assert main_template["attributes"]["parent"] == "hello-artifacts"
        assert "container_image" not in main_template["attributes"]

        # Template node: produce (has container image)
        produce_template = next(
            (n for n in nodes if n["id"] == "argo://kg-builder/WorkflowTemplate/hello-artifacts/produce"),
            None,
        )
        assert produce_template is not None
        assert produce_template["label"] == "produce"
        assert produce_template["file_type"] == "argo"
        assert produce_template["source_file"] == str(wftmpl)
        assert produce_template["source_location"] == "doc0"
        assert produce_template["attributes"]["template"] == "produce"
        assert produce_template["attributes"]["parent"] == "hello-artifacts"
        assert produce_template["attributes"]["container_image"] == "alpine:3.20"


def test_extract_argo_emits_invokes_edges_for_dag_tasks_and_steps():
    """Argo execution templates (dag and steps) emit invokes edges from the
    containing template node to each referenced template node. Resolved refs
    produce EXTRACTED confidence; unresolved refs produce AMBIGUOUS edges
    pointing at a deduplicated placeholder node."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wftmpl = td_path / "invokes-test.yaml"
        wftmpl.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: test
spec:
  templates:
    - name: main
      dag:
        tasks:
          - name: produce-task
            template: produce
          - name: consume-task
            template: consume
          - name: missing-task
            template: missing
    - name: produce
      container:
        image: alpine:3.20
    - name: consume
      steps:
        - - name: c1
            template: produce
""",
            encoding="utf-8",
        )

        result = extract_k8s(wftmpl)

        edges = result["edges"]
        nodes = result["nodes"]

        assert len(edges) == 4, f"Expected 4 edges, got {len(edges)}: {edges}"

        main_id = "argo://_cluster/WorkflowTemplate/test/main"
        produce_id = "argo://_cluster/WorkflowTemplate/test/produce"
        consume_id = "argo://_cluster/WorkflowTemplate/test/consume"
        missing_id = "argo://_cluster/WorkflowTemplate/test/missing#unresolved"

        expected = [
            {
                "source": main_id,
                "target": produce_id,
                "relation": "invokes",
                "confidence": "EXTRACTED",
                "source_file": str(wftmpl),
            },
            {
                "source": main_id,
                "target": consume_id,
                "relation": "invokes",
                "confidence": "EXTRACTED",
                "source_file": str(wftmpl),
            },
            {
                "source": consume_id,
                "target": produce_id,
                "relation": "invokes",
                "confidence": "EXTRACTED",
                "source_file": str(wftmpl),
            },
            {
                "source": main_id,
                "target": missing_id,
                "relation": "invokes",
                "confidence": "AMBIGUOUS",
                "source_file": str(wftmpl),
            },
        ]

        for exp in expected:
            assert any(e == exp for e in edges), f"Missing expected edge {exp!r}"

        # Placeholder node for unresolved template
        ph_nodes = [n for n in nodes if n["id"] == missing_id]
        assert len(ph_nodes) == 1
        assert ph_nodes[0] == {
            "id": missing_id,
            "label": "missing (unresolved)",
            "file_type": "argo",
            "source_file": str(wftmpl),
            "attributes": {"unresolved": True},
        }


def test_extract_argo_collects_workflow_template_ref_candidates():
    """Argo CronWorkflow and Workflow manifests with spec.workflowSpec.workflowTemplateRef
    or clusterWorkflowTemplateRef emit argo_candidates side-channel entries carrying the
    referencing workflow node id as source, target kind, namespace, and source_file.
    No references edge is emitted at pass 1."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # CronWorkflow with namespaced workflowTemplateRef
        cron = td_path / "cron.yaml"
        cron.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: hourly-etl
  namespace: payments
spec:
  workflowSpec:
    workflowTemplateRef:
      name: daily-env-cost-ingestion
""",
            encoding="utf-8",
        )

        # Workflow with clusterWorkflowTemplateRef (no metadata.namespace → _cluster)
        cluster = td_path / "cluster.yaml"
        cluster.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: cluster-pipeline
spec:
  workflowSpec:
    clusterWorkflowTemplateRef:
      name: shared-data-template
""",
            encoding="utf-8",
        )

        result_cron = extract_k8s(cron)
        result_cluster = extract_k8s(cluster)

        # --- CronWorkflow: namespaced workflowTemplateRef ---
        assert "argo_candidates" in result_cron
        cands_cron = result_cron["argo_candidates"]
        assert len(cands_cron) == 1
        assert cands_cron[0] == {
            "source": "argo://payments/CronWorkflow/hourly-etl",
            "target_name": "daily-env-cost-ingestion",
            "target_kind": "WorkflowTemplate",
            "namespace": "payments",
            "source_file": str(cron),
            "relation": "references",
        }

        # --- Workflow: clusterWorkflowTemplateRef ---
        assert "argo_candidates" in result_cluster
        cands_cluster = result_cluster["argo_candidates"]
        assert len(cands_cluster) == 1
        assert cands_cluster[0] == {
            "source": "argo://_cluster/Workflow/cluster-pipeline",
            "target_name": "shared-data-template",
            "target_kind": "ClusterWorkflowTemplate",
            "namespace": "_cluster",
            "source_file": str(cluster),
            "relation": "references",
        }

        # No references edges emitted at pass 1 (edges stay within-doc only).
        assert all(e.get("relation") != "references" for e in result_cron["edges"])
        assert all(e.get("relation") != "references" for e in result_cluster["edges"])


def test_extract_argo_emits_depends_on_edges_for_dag_task_dependencies():
    """Argo dag tasks with a non-empty dependencies list emit depends_on edges
    from the dependent task's template to each dependency task's template,
    always with INFERRED confidence. The dependency list references sibling
    task names, not template names directly."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wftmpl = td_path / "depends-on-test.yaml"
        wftmpl.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: pipeline
  namespace: data-team
spec:
  templates:
    - name: main
      dag:
        tasks:
          - name: produce-task
            template: produce
          - name: consume-task
            template: consume
            dependencies:
              - produce-task
    - name: produce
      container:
        image: alpine:3.20
    - name: consume
      container:
        image: alpine:3.20
""",
            encoding="utf-8",
        )

        result = extract_k8s(wftmpl)
        edges = result["edges"]

        # The sole depends_on edge: consume template -> produce template
        depends_on_edges = [e for e in edges if e.get("relation") == "depends_on"]
        assert len(depends_on_edges) == 1, (
            f"Expected exactly 1 depends_on edge, got {len(depends_on_edges)}: {depends_on_edges}"
        )

        edge = depends_on_edges[0]
        assert edge == {
            "source": "argo://data-team/WorkflowTemplate/pipeline/consume",
            "target": "argo://data-team/WorkflowTemplate/pipeline/produce",
            "relation": "depends_on",
            "confidence": "INFERRED",
            "source_file": str(wftmpl),
        }

        # Invokes edges from C3 are still produced alongside depends_on.
        invokes_edges = [e for e in edges if e.get("relation") == "invokes"]
        assert len(invokes_edges) == 2, (
            f"Expected 2 invokes edges, got {len(invokes_edges)}: {invokes_edges}"
        )


def test_extract_argo_collects_workflow_template_ref_from_spec_directly_for_workflow():
    """A canonical Workflow manifest (not CronWorkflow) places workflowTemplateRef
    directly under spec, not under spec.workflowSpec. The extractor must fall back
    to reading from spec when workflowSpec is absent and still emit an argo_candidate
    with the correct source, namespace, and target kind."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wf = td_path / "workflow.yaml"
        wf.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: my-wf
  namespace: jobs
spec:
  workflowTemplateRef:
    name: some-template
""",
            encoding="utf-8",
        )

        result = extract_k8s(wf)

        assert "argo_candidates" in result
        cands = result["argo_candidates"]
        assert len(cands) == 1
        assert cands[0] == {
            "source": "argo://jobs/Workflow/my-wf",
            "target_name": "some-template",
            "target_kind": "WorkflowTemplate",
            "namespace": "jobs",
            "source_file": str(wf),
            "relation": "references",
        }


def test_resolve_argo_references_emits_ambiguous_edges_and_deduped_placeholder_for_unresolved_targets():
    """Unresolved argo_candidates emit an AMBIGUOUS references edge to a
    #unresolved id, and exactly one placeholder node per unique unresolved target
    (deduped across candidates, not per-candidate). Two candidates referencing the
    same missing name produce two edges but one node."""
    all_nodes: list[dict] = []
    all_edges: list[dict] = []

    per_file = [
        {
            "argo_candidates": [
                {
                    "source": "argo://payments/Workflow/etl-a",
                    "target_name": "missing-wft",
                    "target_kind": "WorkflowTemplate",
                    "namespace": "payments",
                    "source_file": "a.yaml",
                    "relation": "references",
                },
                {
                    "source": "argo://payments/Workflow/etl-b",
                    "target_name": "missing-wft",
                    "target_kind": "WorkflowTemplate",
                    "namespace": "payments",
                    "source_file": "b.yaml",
                    "relation": "references",
                },
            ]
        }
    ]

    _resolve_argo_references(per_file, all_nodes, all_edges)

    # Two AMBIGUOUS edges, one per candidate, both pointing at the same unresolved target.
    assert len(all_edges) == 2, (
        f"Expected 2 AMBIGUOUS edges, got {len(all_edges)}: {all_edges!r}"
    )
    expected_unresolved = "argo://payments/WorkflowTemplate/missing-wft#unresolved"
    for edge in all_edges:
        assert edge == {
            "source": edge["source"],
            "target": expected_unresolved,
            "relation": "references",
            "confidence": "AMBIGUOUS",
            "source_file": edge["source_file"],
        }
    sources = {e["source"] for e in all_edges}
    assert sources == {
        "argo://payments/Workflow/etl-a",
        "argo://payments/Workflow/etl-b",
    }
    source_files = {e["source_file"] for e in all_edges}
    assert source_files == {"a.yaml", "b.yaml"}

    # Exactly one placeholder node for the shared unresolved target.
    assert len(all_nodes) == 1, (
        f"Expected 1 placeholder node, got {len(all_nodes)}: {all_nodes!r}"
    )
    ph = all_nodes[0]
    assert ph == {
        "id": expected_unresolved,
        "label": "WorkflowTemplate/missing-wft (unresolved)",
        "file_type": "argo",
        "source_file": "a.yaml",
        "attributes": {"unresolved": True},
    }


def test_resolve_argo_references_emits_extracted_references_edges_for_workflow_level_targets():
    """Pass 2 resolution: argo_candidates for workflow-level targets (3-segment argo:// id)
    become references edges with EXTRACTED confidence. Template nodes (4 segments) and
    #unresolved/placeholder ids are excluded from the index so they are never targets."""
    all_nodes = [
        {
            "id": "argo://payments/WorkflowTemplate/daily-env-cost-ingestion",
            "label": "WorkflowTemplate/daily-env-cost-ingestion",
            "file_type": "argo",
            "source_file": "tmpl.yaml",
            "attributes": {"kind": "WorkflowTemplate", "namespace": "payments"},
        },
        {
            "id": "argo://payments/CronWorkflow/cron",
            "label": "CronWorkflow/cron",
            "file_type": "argo",
            "source_file": "cron.yaml",
            "attributes": {"kind": "CronWorkflow", "namespace": "payments"},
        },
        # Template node (4 segments) must NOT be indexed as a resolution target.
        {
            "id": "argo://payments/WorkflowTemplate/daily-env-cost-ingestion/main",
            "label": "main",
            "file_type": "argo",
            "source_file": "tmpl.yaml",
            "attributes": {"template": "main", "parent": "daily-env-cost-ingestion"},
        },
    ]

    per_file = [
        {
            "argo_candidates": [
                {
                    "source": "argo://payments/CronWorkflow/cron",
                    "target_name": "daily-env-cost-ingestion",
                    "target_kind": "WorkflowTemplate",
                    "namespace": "payments",
                    "source_file": "cron.yaml",
                    "relation": "references",
                }
            ]
        }
    ]

    all_edges: list[dict] = []
    _resolve_argo_references(per_file, all_nodes, all_edges)

    assert len(all_edges) == 1
    assert all_edges[0] == {
        "source": "argo://payments/CronWorkflow/cron",
        "target": "argo://payments/WorkflowTemplate/daily-env-cost-ingestion",
        "relation": "references",
        "confidence": "EXTRACTED",
        "source_file": "cron.yaml",
    }


def test_extract_k8s_is_deterministic_for_argo_workflow_template():
    """Two extract_k8s runs over an unchanged Argo WorkflowTemplate must produce
    byte-identical node and edge dicts, so downstream build and push steps are
    stable across re-runs."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wftmpl = td_path / "determinism-test.yaml"
        wftmpl.write_text(
            """\
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: pipeline
  namespace: data-team
spec:
  entrypoint: main
  workflowTemplateRef:
    name: shared-template
  templates:
    - name: main
      dag:
        tasks:
          - name: produce-task
            template: produce
          - name: consume-task
            template: consume
            dependencies:
              - produce-task
    - name: produce
      container:
        image: alpine:3.20
    - name: consume
      container:
        image: alpine:3.20
""",
            encoding="utf-8",
        )

        result1 = extract_k8s(wftmpl)
        result2 = extract_k8s(wftmpl)

        assert result1 == result2


def test_extract_k8s_emits_kustomize_node_for_kustomization_manifest(tmp_path):
    """A kustomization.yaml with apiVersion kustomize.config.k8s.io/v1beta1 and
    kind Kustomization is routed to _extract_kustomize, which emits a single node
    with kustomize:// id, file_type='kustomize', and attributes carrying the
    resources list, namespace, and dir path. No edges are generated."""
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: gds-api
resources:
  - ../base
  - ./api
""",
        encoding="utf-8",
    )

    result = extract_k8s(kustomization)

    nodes = result["nodes"]
    assert len(nodes) == 1, f"Expected 1 node, got {len(nodes)}: {nodes!r}"
    node = nodes[0]

    dirname = tmp_path.as_posix()
    assert node["id"] == f"kustomize://{dirname}/kustomization.yaml"
    assert node["label"] == "kustomization.yaml"
    assert node["file_type"] == "kustomize"
    assert node["source_file"] == str(kustomization)
    assert node["source_location"] == "doc0"
    assert node["attributes"]["dir"] == dirname
    assert node["attributes"]["resources"] == ["../base", "./api"]
    assert node["attributes"]["namespace"] == "gds-api"

    assert result["edges"] == []


def test_extract_kustomize_emits_generated_nodes_and_generates_edges(tmp_path):
    """A kustomization with configMapGenerator and secretGenerator entries yields
    one ConfigMap node per configMapGenerator entry and one Secret node per
    secretGenerator entry, each with a k8s:// id, plus a generates edge from the
    kustomization node to each generated node with EXTRACTED confidence. The
    generator namespace is used, not the kustomization namespace."""
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: gds-api
resources:
  - ../base
configMapGenerator:
  - name: gds-api-session-sizing
    namespace: overlay
  - name: gds-api-config
secretGenerator:
  - name: gds-api-secret
""",
        encoding="utf-8",
    )

    result = extract_k8s(kustomization)

    nodes = result["nodes"]
    edges = result["edges"]
    dirname = tmp_path.as_posix()
    kustomize_id = f"kustomize://{dirname}/kustomization.yaml"

    # Total nodes = 1 kustomization + 2 configMapGenerator entries + 1 secretGenerator entry
    assert len(nodes) == 4, f"Expected 4 nodes, got {len(nodes)}: {nodes!r}"

    # Kustomization node still present
    kustomize_node = next(n for n in nodes if n["id"] == kustomize_id)
    assert kustomize_node is not None

    # Generated ConfigMap nodes
    cm1 = next(n for n in nodes if n["id"] == "k8s://overlay/ConfigMap/gds-api-session-sizing")
    assert cm1 == {
        "id": "k8s://overlay/ConfigMap/gds-api-session-sizing",
        "label": "ConfigMap/gds-api-session-sizing",
        "file_type": "k8s",
        "source_file": str(kustomization),
        "source_location": "doc0",
        "attributes": {
            "kind": "ConfigMap",
            "generated_by": kustomize_id,
            "generator": "configMapGenerator",
            "name": "gds-api-session-sizing",
        },
    }

    cm2 = next(n for n in nodes if n["id"] == "k8s://_cluster/ConfigMap/gds-api-config")
    assert cm2 == {
        "id": "k8s://_cluster/ConfigMap/gds-api-config",
        "label": "ConfigMap/gds-api-config",
        "file_type": "k8s",
        "source_file": str(kustomization),
        "source_location": "doc0",
        "attributes": {
            "kind": "ConfigMap",
            "generated_by": kustomize_id,
            "generator": "configMapGenerator",
            "name": "gds-api-config",
        },
    }

    # Generated Secret node (no namespace → defaults to _cluster)
    sec = next(n for n in nodes if n["id"] == "k8s://_cluster/Secret/gds-api-secret")
    assert sec == {
        "id": "k8s://_cluster/Secret/gds-api-secret",
        "label": "Secret/gds-api-secret",
        "file_type": "k8s",
        "source_file": str(kustomization),
        "source_location": "doc0",
        "attributes": {
            "kind": "Secret",
            "generated_by": kustomize_id,
            "generator": "secretGenerator",
            "name": "gds-api-secret",
        },
    }

    # generates edges: one per generated node, always EXTRACTED
    assert len(edges) == 3, f"Expected 3 edges, got {len(edges)}: {edges!r}"

    expected_edges = [
        {
            "source": kustomize_id,
            "target": "k8s://overlay/ConfigMap/gds-api-session-sizing",
            "relation": "generates",
            "confidence": "EXTRACTED",
            "source_file": str(kustomization),
        },
        {
            "source": kustomize_id,
            "target": "k8s://_cluster/ConfigMap/gds-api-config",
            "relation": "generates",
            "confidence": "EXTRACTED",
            "source_file": str(kustomization),
        },
        {
            "source": kustomize_id,
            "target": "k8s://_cluster/Secret/gds-api-secret",
            "relation": "generates",
            "confidence": "EXTRACTED",
            "source_file": str(kustomization),
        },
    ]
    for exp in expected_edges:
        assert any(e == exp for e in edges), f"Missing expected edge {exp!r}"


def test_extract_kustomize_stashes_kustomize_candidates_for_string_entries(tmp_path):
    """A kustomization with resources, bases, and components lists emits one
    kustomize_candidate per string entry, shaped with source, target_path, dir,
    source_file, and relation 'includes'. Non-string entries are skipped."""
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: gds-api
resources:
  - ../base
  - ./api
bases:
  - ../../shared
components:
  - ./components/auth
  - {not: a string}
""",
        encoding="utf-8",
    )

    result = extract_k8s(kustomization)

    dirname = tmp_path.as_posix()
    kustomize_id = f"kustomize://{dirname}/kustomization.yaml"

    candidates = result["kustomize_candidates"]
    assert candidates == [
        {
            "source": kustomize_id,
            "target_path": "../base",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "./api",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "../../shared",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "./components/auth",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
    ]


def test_resolve_kustomize_includes_emits_extracted_edge_for_resolved_path_and_no_edge_for_unresolved():
    """Pass 2 resolution: a kustomize_candidate whose resolved path matches a node
    indexed by (source_dir, basename) emits an includes edge with EXTRACTED
    confidence. A candidate whose resolved path does NOT match is not emitted as
    EXTRACTED (that branch is AMBIGUOUS, handled separately)."""
    kustomize_id = "kustomize://overlays/prod/kustomization.yaml"
    kustomize_source = "overlays/prod/kustomization.yaml"

    all_nodes = [
        {
            "id": kustomize_id,
            "label": "kustomization.yaml",
            "file_type": "k8s",
            "source_file": kustomize_source,
            "source_location": "doc0",
            "attributes": {"dir": "overlays/prod", "namespace": "prod"},
        },
        {
            "id": "k8s://default/Deployment/api",
            "label": "Deployment/api",
            "file_type": "k8s",
            "source_file": "overlays/base/deployment.yaml",
            "source_location": "doc0",
            "attributes": {"kind": "Deployment", "namespace": "default"},
        },
    ]

    per_file = [
        {
            "nodes": [],
            "edges": [],
            "k8s_candidates": [],
            "kustomize_candidates": [
                # Resolves to overlays/base/deployment.yaml → matches the k8s node.
                {
                    "source": kustomize_id,
                    "target_path": "../base/deployment.yaml",
                    "dir": "overlays/prod",
                    "source_file": kustomize_source,
                    "relation": "includes",
                },
                # Resolves to overlays/prod/missing.yaml → no match.
                {
                    "source": kustomize_id,
                    "target_path": "./missing.yaml",
                    "dir": "overlays/prod",
                    "source_file": kustomize_source,
                    "relation": "includes",
                },
            ],
        }
    ]

    all_edges: list[dict] = []
    _resolve_kustomize_includes(per_file, all_nodes, all_edges)

    extracted = [e for e in all_edges if e.get("confidence") == "EXTRACTED"]
    assert len(extracted) == 1, f"Expected 1 EXTRACTED edge, got {len(extracted)}: {extracted!r}"
    assert extracted[0] == {
        "source": kustomize_id,
        "target": "k8s://default/Deployment/api",
        "relation": "includes",
        "confidence": "EXTRACTED",
        "source_file": kustomize_source,
    }

    # No EXTRACTED edge was emitted for the unresolved candidate (AMBIGUOUS is C4).


def test_extract_kustomize_skips_remote_url_candidates(tmp_path):
    """Remote URL entries in resources, bases, and components are skipped so
    they never become candidates that would later resolve into bogus AMBIGUOUS
    placeholder joins. Only local paths produce kustomize_candidates."""
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: gds-api
resources:
  - https://example.com/remote.yaml
  - ../base
  - ./api
  - git::https://github.com/org/repo.git//path
  - ssh://host/path
  - ./local
bases:
  - http://example.com/base.yaml
  - ../../shared
components:
  - https://component.example.com/
  - ./components/auth
""",
        encoding="utf-8",
    )

    result = extract_k8s(kustomization)

    dirname = tmp_path.as_posix()
    kustomize_id = f"kustomize://{dirname}/kustomization.yaml"

    candidates = result["kustomize_candidates"]
    assert candidates == [
        {
            "source": kustomize_id,
            "target_path": "../base",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "./api",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "./local",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "../../shared",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
        {
            "source": kustomize_id,
            "target_path": "./components/auth",
            "dir": dirname,
            "source_file": str(kustomization),
            "relation": "includes",
        },
    ]


def test_resolve_kustomize_includes_excludes_generated_nodes_from_index():
    """A generated ConfigMap/Secret node shares the same source_file as its parent
    kustomization node and has a k8s:// id with no '#'. If the index does not
    exclude generated nodes, the generated node overwrites the kustomization
    node (last-write-wins), causing an 'includes' edge to target the generated
    resource instead of the kustomization file. The index must skip nodes that
    have both generator and generated_by attributes."""
    kustomize_id = "kustomize://base/kustomization.yaml"
    kustomize_source = "base/kustomization.yaml"

    all_nodes = [
        # The kustomization node (must come first to show the bug — generated
        # node coming AFTER overwrites the index entry).
        {
            "id": kustomize_id,
            "label": "kustomization.yaml",
            "file_type": "k8s",
            "source_file": kustomize_source,
            "source_location": "doc0",
            "attributes": {"dir": "base", "namespace": "_cluster"},
        },
        # The generated ConfigMap node — same source_file, k8s:// id, no '#'.
        # In the current bug this overwrites the kustomization node in the index.
        {
            "id": "k8s://_cluster/ConfigMap/foo",
            "label": "ConfigMap/foo",
            "file_type": "k8s",
            "source_file": kustomize_source,
            "source_location": "doc0",
            "attributes": {
                "kind": "ConfigMap",
                "generated_by": kustomize_id,
                "generator": "configMapGenerator",
                "name": "foo",
            },
        },
    ]

    per_file = [
        {
            "nodes": [],
            "edges": [],
            "k8s_candidates": [],
            "kustomize_candidates": [
                {
                    "source": "kustomize://overlays/prod/kustomization.yaml",
                    "target_path": "../../base/kustomization.yaml",
                    "dir": "overlays/prod",
                    "source_file": "overlays/prod/kustomization.yaml",
                    "relation": "includes",
                },
            ],
        }
    ]

    all_edges: list[dict] = []
    _resolve_kustomize_includes(per_file, all_nodes, all_edges)

    extracted = [e for e in all_edges if e.get("confidence") == "EXTRACTED"]
    assert len(extracted) == 1, f"Expected 1 EXTRACTED edge, got {len(extracted)}: {extracted!r}"
    assert extracted[0] == {
        "source": "kustomize://overlays/prod/kustomization.yaml",
        "target": kustomize_id,
        "relation": "includes",
        "confidence": "EXTRACTED",
        "source_file": "overlays/prod/kustomization.yaml",
    }


def test_resolve_kustomize_includes_ambiguous_unresolved_paths():
    """Unresolved kustomize includes emit AMBIGUOUS edges with deduplicated
    placeholder nodes.

    Two different missing paths produce two AMBIGUOUS edges and two placeholder
    nodes. Two candidates referencing the same missing path produce two edges
    but only one placeholder node (deduplicated by the resolved path).

    Placeholder nodes use file_type 'kustomize' matching their kustomize:// id
    scheme, consistent with argo/k8s placeholder conventions.
    """
    all_nodes: list[dict] = []
    all_edges: list[dict] = []

    per_file = [
        {
            "nodes": [],
            "edges": [],
            "k8s_candidates": [],
            "kustomize_candidates": [
                # Two different unresolved paths
                {
                    "source": "kustomize://overlays/prod/kustomization.yaml",
                    "target_path": "./missing-a.yaml",
                    "dir": "overlays/prod",
                    "source_file": "overlays/prod/kustomization.yaml",
                    "relation": "includes",
                },
                {
                    "source": "kustomize://overlays/prod/kustomization.yaml",
                    "target_path": "./missing-b.yaml",
                    "dir": "overlays/prod",
                    "source_file": "overlays/prod/kustomization.yaml",
                    "relation": "includes",
                },
                # Same missing path as the first — should dedup placeholder node
                {
                    "source": "kustomize://overlays/prod/kustomization.yaml",
                    "target_path": "./missing-a.yaml",
                    "dir": "overlays/prod",
                    "source_file": "overlays/prod/kustomization.yaml",
                    "relation": "includes",
                },
            ],
        }
    ]

    _resolve_kustomize_includes(per_file, all_nodes, all_edges)

    ambiguous = [e for e in all_edges if e.get("confidence") == "AMBIGUOUS"]
    assert len(ambiguous) == 3, (
        f"Expected 3 AMBIGUOUS edges, got {len(ambiguous)}: {ambiguous!r}"
    )

    placeholder_nodes = [
        n for n in all_nodes if n.get("id", "").endswith("#unresolved")
    ]
    assert len(placeholder_nodes) == 2, (
        f"Expected 2 placeholder nodes, got {len(placeholder_nodes)}: {placeholder_nodes!r}"
    )

    missing_a_id = "kustomize://overlays/prod/missing-a.yaml#unresolved"
    missing_a_node = next(n for n in placeholder_nodes if n["id"] == missing_a_id)
    assert missing_a_node == {
        "id": missing_a_id,
        "label": "missing-a.yaml (unresolved)",
        "file_type": "kustomize",
        "source_file": "overlays/prod/kustomization.yaml",
        "attributes": {"unresolved": True},
    }

    missing_b_id = "kustomize://overlays/prod/missing-b.yaml#unresolved"
    missing_b_node = next(n for n in placeholder_nodes if n["id"] == missing_b_id)
    assert missing_b_node == {
        "id": missing_b_id,
        "label": "missing-b.yaml (unresolved)",
        "file_type": "kustomize",
        "source_file": "overlays/prod/kustomization.yaml",
        "attributes": {"unresolved": True},
    }

    expected_edges = [
        {
            "source": "kustomize://overlays/prod/kustomization.yaml",
            "target": missing_a_id,
            "relation": "includes",
            "confidence": "AMBIGUOUS",
            "source_file": "overlays/prod/kustomization.yaml",
        },
        {
            "source": "kustomize://overlays/prod/kustomization.yaml",
            "target": missing_b_id,
            "relation": "includes",
            "confidence": "AMBIGUOUS",
            "source_file": "overlays/prod/kustomization.yaml",
        },
        {
            "source": "kustomize://overlays/prod/kustomization.yaml",
            "target": missing_a_id,
            "relation": "includes",
            "confidence": "AMBIGUOUS",
            "source_file": "overlays/prod/kustomization.yaml",
        },
    ]
    for exp in expected_edges:
        assert any(e == exp for e in ambiguous), f"Missing expected edge {exp!r}"


def test_kustomize_file_type_passes_validation():
    """A node with file_type='kustomize' must not trigger a validation error."""
    from graphify.validate import validate_extraction

    extraction = {
        "nodes": [
            {
                "id": "kustomize://base/kustomization.yaml",
                "label": "kustomization",
                "file_type": "kustomize",
                "source_file": "kustomization.yaml",
            }
        ],
        "edges": [],
    }

    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'kustomize'" in e]
    assert file_type_errors == []


def test_extract_k8s_emits_image_node_and_runs_edge_for_concrete_container_image():
    """A Deployment with a concrete container image emits an image node and a runs
    edge from the workload to the image. The image node id is the tag-less registry
    path, file_type is 'image', and source_file is None."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        deployment = td_path / "deployment.yaml"
        deployment.write_text(
            """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: payments
spec:
  template:
    spec:
      containers:
        - name: app
          image: europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility:806406f5-unclean
""",
            encoding="utf-8",
        )

        result = extract_k8s(deployment)

        nodes = result["nodes"]
        edges = result["edges"]

        # Workload node
        workload = next(
            n for n in nodes if n["id"] == "k8s://payments/Deployment/api-server"
        )
        assert workload is not None

        # Image node
        image_id = "image://europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility"
        image_node = next((n for n in nodes if n["id"] == image_id), None)
        assert image_node is not None
        assert image_node == {
            "id": image_id,
            "label": "europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility",
            "file_type": "image",
            "source_file": None,
            "attributes": {
                "registry": "europe-west1-docker.pkg.dev",
                "tags": ["806406f5-unclean"],
            },
        }

        # Exactly one runs edge
        assert len(edges) == 1
        assert edges[0] == {
            "source": "k8s://payments/Deployment/api-server",
            "target": image_id,
            "relation": "runs",
            "confidence": "EXTRACTED",
            "source_file": str(deployment),
        }


def test_extract_k8s_skips_placeholder_image_and_emits_no_runs_edge():
    """A Pod with a placeholder image value (_IMAGE) is skipped: no image node and
    no runs edge are emitted, and edges remains empty."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pod = td_path / "pod.yaml"
        pod.write_text(
            """\
apiVersion: v1
kind: Pod
metadata:
  name: worker
  namespace: jobs
spec:
  containers:
    - name: main
      image: _IMAGE
""",
            encoding="utf-8",
        )

        result = extract_k8s(pod)

        # No image node should exist
        assert all(not n["id"].startswith("image://") for n in result["nodes"])

        # edges must stay empty
        assert result["edges"] == []


def test_extract_k8s_dedups_image_nodes_and_runs_edges_for_same_workload_same_image_path():
    """A Deployment with two containers referencing the same registry path with
    different tags emits only one image node and one runs edge, because the
    tag-less image id collapses both references."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        deployment = td_path / "deployment.yaml"
        deployment.write_text(
            """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: payments
spec:
  template:
    spec:
      containers:
        - name: app
          image: repo/img:v1
        - name: sidecar
          image: repo/img:v2
""",
            encoding="utf-8",
        )

        result = extract_k8s(deployment)

        nodes = result["nodes"]
        edges = result["edges"]

        # Exactly one workload node and one image node
        assert len(nodes) == 2
        workload_nodes = [n for n in nodes if n["id"] == "k8s://payments/Deployment/api-server"]
        image_nodes = [n for n in nodes if n["id"].startswith("image://")]
        assert len(workload_nodes) == 1
        assert len(image_nodes) == 1

        image_node = image_nodes[0]
        assert image_node == {
            "id": "image://repo/img",
            "label": "repo/img",
            "file_type": "image",
            "source_file": None,
            "attributes": {
                "registry": "repo",
                "tags": ["v1"],
            },
        }

        # Exactly one runs edge
        assert len(edges) == 1
        assert edges[0] == {
            "source": "k8s://payments/Deployment/api-server",
            "target": "image://repo/img",
            "relation": "runs",
            "confidence": "EXTRACTED",
            "source_file": str(deployment),
        }


def test_extract_k8s_dedups_image_nodes_and_emits_one_runs_edge_per_workload():
    """A multi-document manifest with two distinct workloads referencing the same
    image emits one image node and two runs edges (one per workload)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = td_path / "manifest.yaml"
        manifest.write_text(
            """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: default
spec:
  template:
    spec:
      containers:
        - name: app
          image: other/registry:tag
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: backend
  namespace: default
spec:
  template:
    spec:
      containers:
        - name: worker
          image: other/registry:tag
""",
            encoding="utf-8",
        )

        result = extract_k8s(manifest)

        nodes = result["nodes"]
        edges = result["edges"]

        # Two workload nodes and one image node
        assert len(nodes) == 3
        workload_nodes = [n for n in nodes if n["file_type"] == "k8s"]
        image_nodes = [n for n in nodes if n["id"].startswith("image://")]
        assert len(workload_nodes) == 2
        assert len(image_nodes) == 1
        assert image_nodes[0]["id"] == "image://other/registry"

        # Two runs edges, one per workload
        assert len(edges) == 2
        expected_edges = [
            {
                "source": "k8s://default/Deployment/frontend",
                "target": "image://other/registry",
                "relation": "runs",
                "confidence": "EXTRACTED",
                "source_file": str(manifest),
            },
            {
                "source": "k8s://default/StatefulSet/backend",
                "target": "image://other/registry",
                "relation": "runs",
                "confidence": "EXTRACTED",
                "source_file": str(manifest),
            },
        ]
        for exp in expected_edges:
            assert any(e == exp for e in edges), f"Missing expected edge {exp!r}"


def test_extract_k8s_is_deterministic_for_kustomization(tmp_path):
    """Two extract_k8s runs over an unchanged kustomization must produce
    byte-identical node and edge dicts, so downstream build and push steps are
    stable across re-runs."""
    kustomization = tmp_path / "kustomization.yaml"
    kustomization.write_text(
        """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: gds-api
resources:
  - ../base
  - ./api
bases:
  - ../../shared
components:
  - ./components/auth
configMapGenerator:
  - name: gds-api-config
    namespace: overlay
  - name: gds-api-defaults
secretGenerator:
  - name: gds-api-secret
""",
        encoding="utf-8",
    )

    result1 = extract_k8s(kustomization)
    result2 = extract_k8s(kustomization)

    assert result1 == result2


def test_extract_k8s_routes_ci_workflow_to_extract_ci_and_emits_job_nodes(tmp_path):
    """A GitHub Actions workflow with two jobs is routed through _extract_ci,
    emitting one node per job with file_type='ci', label set to the job key,
    and a ci:// id that is stable and /-joined. No edges or ci_candidates are
    emitted at pass 1."""
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """\
name: CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
""",
        encoding="utf-8",
    )

    result = extract_k8s(workflow)

    nodes = result["nodes"]
    edges = result["edges"]
    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}: {nodes!r}"
    assert edges == []
    assert "ci_candidates" not in result or result.get("ci_candidates", []) == []

    labels = {n["label"] for n in nodes}
    assert labels == {"build", "deploy"}

    for n in nodes:
        assert n["file_type"] == "ci"
        assert n["source_file"] == str(workflow)
        assert n["source_location"] == "doc0"
        assert n["id"].startswith("ci://")
        assert n["label"] in n["id"]


def test_extract_ci_defensive_jobs_not_dict_returns_empty():
    """A CI workflow where jobs is not a dict yields zero nodes and zero edges."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        workflow = td_path / "workflow.yaml"
        workflow.write_text(
            """\
name: CI
on: push
jobs: just-a-string
""",
            encoding="utf-8",
        )

        result = extract_k8s(workflow)
        assert result["nodes"] == []
        assert result["edges"] == []
