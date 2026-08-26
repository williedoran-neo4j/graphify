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
