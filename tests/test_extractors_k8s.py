"""Tests for the Kubernetes YAML extractor (graphify/extractors/k8s.py)."""
from __future__ import annotations

from pathlib import Path

from graphify.extractors.k8s import (
    Dialect,
    _resolve_k8s_references,
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

        # Non-k8s must return empty.
        assert result_non["nodes"] == []
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

