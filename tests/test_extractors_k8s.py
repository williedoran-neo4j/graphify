"""Tests for the Kubernetes YAML extractor (graphify/extractors/k8s.py)."""
from __future__ import annotations

from pathlib import Path

from graphify.extractors.k8s import Dialect, detect_dialect


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
