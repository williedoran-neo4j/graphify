"""Kubernetes YAML extractor — dialect detection and node extraction."""
from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import yaml


class Dialect(Enum):
    K8S_MANIFEST = auto()


def detect_dialect(path: Path, raw_text: str) -> Dialect | None:
    """Classify raw YAML text as a Kubernetes manifest dialect, or None."""
    try:
        docs = list(yaml.safe_load_all(raw_text))
    except yaml.YAMLError:
        return None
    if not docs:
        return None
    for doc in docs:
        if not isinstance(doc, dict):
            return None
        api_version = doc.get("apiVersion")
        kind = doc.get("kind")
        if not isinstance(api_version, str) or not isinstance(kind, str):
            return None
        if api_version.startswith("argoproj.io/"):
            return None
    return Dialect.K8S_MANIFEST


def extract_k8s(path: Path) -> dict:
    """Extract Kubernetes manifest nodes from a YAML file.

    One node per top-level manifest document, with a raw k8s:// id and
    attributes drawn from metadata/spec. Non-k8s YAML yields no nodes.
    """
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    if detect_dialect(path, raw_text) is not Dialect.K8S_MANIFEST:
        return {"nodes": [], "edges": []}
    nodes = []
    for i, doc in enumerate(yaml.safe_load_all(raw_text)):
        metadata = doc.get("metadata") or {}
        kind = doc.get("kind", "")
        name = metadata.get("name", "")
        namespace = metadata.get("namespace") or "_cluster"
        attributes = {"kind": kind, "namespace": namespace}
        containers = _container_names(doc)
        if containers:
            attributes["containers"] = containers
        nodes.append(
            {
                "id": f"k8s://{namespace}/{kind}/{name}",
                "label": f"{kind}/{name}",
                "file_type": "k8s",
                "source_file": str(path),
                "source_location": f"doc{i}",
                "attributes": attributes,
            }
        )
    return {"nodes": nodes, "edges": []}


def _container_names(doc: dict) -> list[str]:
    """Return container names for Deployment-like and Pod manifests."""
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return []
    containers = []
    template = spec.get("template")
    if isinstance(template, dict) and isinstance(template.get("spec"), dict):
        containers = template["spec"].get("containers")
    elif isinstance(spec.get("containers"), list):
        containers = spec["containers"]
    if not isinstance(containers, list):
        return []
    return [c["name"] for c in containers if isinstance(c, dict) and isinstance(c.get("name"), str)]
