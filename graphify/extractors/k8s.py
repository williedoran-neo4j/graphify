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
    """Extract Kubernetes manifest nodes and candidate references from a YAML file.

    One node per top-level manifest document, with a raw k8s:// id and
    attributes drawn from metadata/spec. k8s_candidates holds one bare dict
    per declared reference (envFrom, env.valueFrom, volumes, serviceAccountName).
    Non-k8s YAML yields no nodes and no candidates.
    """
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    if detect_dialect(path, raw_text) is not Dialect.K8S_MANIFEST:
        return {"nodes": [], "edges": [], "k8s_candidates": []}
    nodes = []
    candidates = []
    for i, doc in enumerate(yaml.safe_load_all(raw_text)):
        metadata = doc.get("metadata") or {}
        kind = doc.get("kind", "")
        name = metadata.get("name", "")
        namespace = metadata.get("namespace") or "_cluster"
        attributes = {"kind": kind, "namespace": namespace}
        containers = _container_names(doc)
        if containers:
            attributes["containers"] = containers
        spec = doc.get("spec")
        if isinstance(spec, dict):
            selector = spec.get("selector")
            if kind == "Service" and isinstance(selector, dict) and selector:
                attributes["selector"] = selector
        labels = None
        if kind != "Service":
            if isinstance(spec, dict):
                template = spec.get("template")
                if isinstance(template, dict):
                    template_metadata = template.get("metadata")
                    if isinstance(template_metadata, dict):
                        template_labels = template_metadata.get("labels")
                        if isinstance(template_labels, dict) and template_labels:
                            labels = template_labels
            if labels is None:
                bare_labels = metadata.get("labels")
                if isinstance(bare_labels, dict) and bare_labels:
                    labels = bare_labels
        if labels is not None:
            attributes["labels"] = labels
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
        base = {
            "namespace": namespace,
            "source_file": str(path),
            "source_kind": kind,
            "source_name": name,
        }
        candidates.extend(_candidates(doc, base))
    return {"nodes": nodes, "edges": [], "k8s_candidates": candidates}


def _pod_spec(doc: dict) -> dict | None:
    """Return the pod-level spec for a workload or Pod manifest, if any."""
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return None
    template = spec.get("template")
    if isinstance(template, dict) and isinstance(template.get("spec"), dict):
        return template["spec"]
    return spec


def _named_ref(holder: dict, key: str, kind: str, base: dict, name_key: str = "name") -> list[dict]:
    """One candidate for a named reference nested inside holder, or none."""
    ref = holder.get(key)
    if isinstance(ref, dict) and isinstance(ref.get(name_key), str):
        return [
            {
                **base,
                "target_name": ref[name_key],
                "target_kind": kind,
                "relation": "references",
            }
        ]
    return []


def _candidates(doc: dict, base: dict) -> list[dict]:
    """Collect candidate references declared by a manifest (candidates, not edges)."""
    out = []
    pod_spec = _pod_spec(doc)
    if pod_spec is None:
        return out
    containers = pod_spec.get("containers")
    if isinstance(containers, list):
        for c in containers:
            if not isinstance(c, dict):
                continue
            env_from = c.get("envFrom")
            if isinstance(env_from, list):
                for item in env_from:
                    if isinstance(item, dict):
                        out.extend(_named_ref(item, "configMapRef", "ConfigMap", base))
                        out.extend(_named_ref(item, "secretRef", "Secret", base))
            env = c.get("env")
            if isinstance(env, list):
                for item in env:
                    if not isinstance(item, dict):
                        continue
                    value_from = item.get("valueFrom")
                    if isinstance(value_from, dict):
                        out.extend(_named_ref(value_from, "configMapKeyRef", "ConfigMap", base))
                        out.extend(_named_ref(value_from, "secretKeyRef", "Secret", base))
    volumes = pod_spec.get("volumes")
    if isinstance(volumes, list):
        for v in volumes:
            if isinstance(v, dict):
                out.extend(_named_ref(v, "configMap", "ConfigMap", base))
                out.extend(_named_ref(v, "secret", "Secret", base, name_key="secretName"))
    sa = pod_spec.get("serviceAccountName")
    if isinstance(sa, str):
        out.append(
            {
                **base,
                "target_name": sa,
                "target_kind": "ServiceAccount",
                "relation": "uses_service_account",
            }
        )
    return out


def _resolve_k8s_references(
    per_file: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Resolve K8s manifest references into edges (pass 2).

    Builds a (namespace, kind, name) -> node_id index from all_nodes and, for
    each candidate in per_file's k8s_candidates, appends a references edge.
    """
    index = {}
    for node in all_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.startswith("k8s://"):
            continue
        namespace, kind, name = node_id[len("k8s://"):].split("/", 2)
        index[(namespace, kind, name)] = node_id
    placeholder_ids = set()
    for entry in per_file:
        for candidate in entry.get("k8s_candidates") or []:
            relation = candidate.get("relation")
            if relation not in ("references", "uses_service_account"):
                continue
            target = index.get(
                (
                    candidate["namespace"],
                    candidate["target_kind"],
                    candidate["target_name"],
                )
            )
            if target is None:
                unresolved_id = (
                    f"k8s://{candidate['namespace']}/"
                    f"{candidate['target_kind']}/{candidate['target_name']}#unresolved"
                )
                all_edges.append(
                    {
                        "source": (
                            f"k8s://{candidate['namespace']}/"
                            f"{candidate['source_kind']}/{candidate['source_name']}"
                        ),
                        "target": unresolved_id,
                        "relation": relation,
                        "confidence": "AMBIGUOUS",
                        "source_file": candidate["source_file"],
                    }
                )
                if unresolved_id not in placeholder_ids:
                    placeholder_ids.add(unresolved_id)
                    all_nodes.append(
                        {
                            "id": unresolved_id,
                            "label": (
                                f"{candidate['target_kind']}/"
                                f"{candidate['target_name']} (unresolved)"
                            ),
                            "file_type": "k8s",
                            "source_file": candidate["source_file"],
                            "attributes": {"unresolved": True},
                        }
                    )
                continue
            all_edges.append(
                {
                    "source": (
                        f"k8s://{candidate['namespace']}/"
                        f"{candidate['source_kind']}/{candidate['source_name']}"
                    ),
                    "target": target,
                    "relation": relation,
                    "confidence": "EXTRACTED",
                    "source_file": candidate["source_file"],
                }
            )


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
