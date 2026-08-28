"""Kubernetes YAML extractor — dialect detection and node extraction."""
from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import yaml


class Dialect(Enum):
    K8S_MANIFEST = auto()
    ARGO_WORKFLOW = auto()
    KUSTOMIZATION = auto()


def detect_dialect(path: Path, raw_text: str) -> Dialect | None:
    """Classify raw YAML text as a Kubernetes manifest dialect, or None."""
    try:
        docs = list(yaml.safe_load_all(raw_text))
    except yaml.YAMLError:
        return None
    if not docs:
        return None
    argo_kinds = {
        "Workflow",
        "WorkflowTemplate",
        "ClusterWorkflowTemplate",
        "CronWorkflow",
    }
    saw_argo = False
    all_argo = True
    saw_kustomize = False
    all_kustomize = True
    for doc in docs:
        if not isinstance(doc, dict):
            return None
        api_version = doc.get("apiVersion")
        kind = doc.get("kind")
        if not isinstance(api_version, str) or not isinstance(kind, str):
            return None
        if api_version.startswith("argoproj.io/"):
            if kind not in argo_kinds:
                return None
            saw_argo = True
            all_kustomize = False
        elif api_version.startswith("kustomize.config.k8s.io/"):
            if kind != "Kustomization":
                return None
            saw_kustomize = True
            all_argo = False
        else:
            all_argo = False
            all_kustomize = False
    if all_argo:
        return Dialect.ARGO_WORKFLOW
    if all_kustomize:
        return Dialect.KUSTOMIZATION
    if saw_argo or saw_kustomize:
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
    dialect = detect_dialect(path, raw_text)
    if dialect is Dialect.ARGO_WORKFLOW:
        return _extract_argo(path, raw_text)
    if dialect is Dialect.KUSTOMIZATION:
        return _extract_kustomize(path, raw_text)
    if dialect is not Dialect.K8S_MANIFEST:
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


def _extract_argo(path: Path, raw_text: str) -> dict:
    """Extract Argo workflow, template nodes, and within-doc invokes edges."""
    nodes = []
    edges = []
    argo_candidates = []
    for i, doc in enumerate(yaml.safe_load_all(raw_text)):
        if not isinstance(doc, dict):
            continue
        metadata = doc.get("metadata") or {}
        kind = doc.get("kind", "")
        name = metadata.get("name", "")
        namespace = metadata.get("namespace") or "_cluster"
        source = {
            "source_file": str(path),
            "source_location": f"doc{i}",
        }
        workflow_id = f"argo://{namespace}/{kind}/{name}"
        workflow_attrs = {"kind": kind, "namespace": namespace}
        spec = doc.get("spec")
        if isinstance(spec, dict):
            entrypoint = spec.get("entrypoint")
            if isinstance(entrypoint, str) and entrypoint:
                workflow_attrs["entrypoint"] = entrypoint
        nodes.append(
            {
                "id": workflow_id,
                "label": f"{kind}/{name}",
                "file_type": "argo",
                **source,
                "attributes": workflow_attrs,
            }
        )
        if isinstance(spec, dict):
            ref_spec = spec.get("workflowSpec")
            if not isinstance(ref_spec, dict):
                ref_spec = spec
            for ref_key, target_kind in (
                ("workflowTemplateRef", "WorkflowTemplate"),
                ("clusterWorkflowTemplateRef", "ClusterWorkflowTemplate"),
            ):
                ref = ref_spec.get(ref_key)
                if not isinstance(ref, dict) or not isinstance(ref.get("name"), str):
                    continue
                ref_name = ref["name"]
                if not ref_name:
                    continue
                ref_ns = namespace if target_kind == "WorkflowTemplate" else "_cluster"
                argo_candidates.append(
                    {
                        "source": workflow_id,
                        "target_name": ref_name,
                        "target_kind": target_kind,
                        "namespace": ref_ns,
                        "source_file": str(path),
                        "relation": "references",
                    }
                )
        templates = spec.get("templates") if isinstance(spec, dict) else None
        if isinstance(templates, list):
            for t in templates:
                if not isinstance(t, dict):
                    continue
                template = t.get("name")
                if not isinstance(template, str) or not template:
                    continue
                template_attrs = {"template": template, "parent": name}
                container_image = None
                container = t.get("container")
                if isinstance(container, dict):
                    container_image = container.get("image")
                script = t.get("script")
                if isinstance(script, dict) and isinstance(script.get("image"), str):
                    container_image = script.get("image")
                if isinstance(container_image, str) and container_image:
                    template_attrs["container_image"] = container_image
                nodes.append(
                    {
                        "id": f"argo://{namespace}/{kind}/{name}/{template}",
                        "label": template,
                        "file_type": "argo",
                        **source,
                        "attributes": template_attrs,
                    }
                )
        if isinstance(templates, list):
            _emit_argo_invokes(
                templates,
                namespace,
                kind,
                name,
                source,
                nodes,
                edges,
            )
    return {"nodes": nodes, "edges": edges, "k8s_candidates": [], "argo_candidates": argo_candidates}


def _emit_argo_invokes(
    templates: list,
    namespace: str,
    kind: str,
    name: str,
    source: dict,
    nodes: list[dict],
    edges: list[dict],
) -> None:
    """Emit invokes edges for dag tasks and steps nested in execution templates.

    Confidences are EXTRACTED when a template ref resolves to a same-doc
    spec.templates[] entry name, and AMBIGUOUS otherwise (a deduplicated
    placeholder node is appended for each unresolved ref).
    """
    template_ids = {
        t["name"]: f"argo://{namespace}/{kind}/{name}/{t['name']}"
        for t in templates
        if isinstance(t, dict) and isinstance(t.get("name"), str)
    }
    placeholders = set()

    def _edge(ref: str, container_id: str) -> None:
        target = template_ids.get(ref)
        if target is not None:
            edges.append(
                {
                    "source": container_id,
                    "target": target,
                    "relation": "invokes",
                    "confidence": "EXTRACTED",
                    "source_file": source["source_file"],
                }
            )
            return
        unresolved_id = f"argo://{namespace}/{kind}/{name}/{ref}#unresolved"
        edges.append(
            {
                "source": container_id,
                "target": unresolved_id,
                "relation": "invokes",
                "confidence": "AMBIGUOUS",
                "source_file": source["source_file"],
            }
        )
        if unresolved_id not in placeholders:
            placeholders.add(unresolved_id)
            nodes.append(
                {
                    "id": unresolved_id,
                    "label": f"{ref} (unresolved)",
                    "file_type": "argo",
                    "source_file": source["source_file"],
                    "attributes": {"unresolved": True},
                }
            )

    for t in templates:
        if not isinstance(t, dict):
            continue
        container_name = t.get("name")
        if not isinstance(container_name, str) or not container_name:
            continue
        container_id = f"argo://{namespace}/{kind}/{name}/{container_name}"
        dag = t.get("dag")
        if isinstance(dag, dict):
            tasks = dag.get("tasks")
            if isinstance(tasks, list):
                task_name_to_template = {
                    task["name"]: task["template"]
                    for task in tasks
                    if isinstance(task, dict)
                    and isinstance(task.get("name"), str)
                    and isinstance(task.get("template"), str)
                    and task["template"]
                }
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    ref = task.get("template")
                    if isinstance(ref, str) and ref:
                        _edge(ref, container_id)
                    this_template = task.get("template")
                    dependencies = task.get("dependencies")
                    if (
                        isinstance(this_template, str)
                        and this_template
                        and isinstance(dependencies, list)
                        and dependencies
                    ):
                        dep_source = f"argo://{namespace}/{kind}/{name}/{this_template}"
                        for dep_name in dependencies:
                            if not isinstance(dep_name, str):
                                continue
                            dep_template = task_name_to_template.get(dep_name)
                            if dep_template is None:
                                continue
                            edges.append(
                                {
                                    "source": dep_source,
                                    "target": (
                                        f"argo://{namespace}/{kind}/{name}/{dep_template}"
                                    ),
                                    "relation": "depends_on",
                                    "confidence": "INFERRED",
                                    "source_file": source["source_file"],
                                }
                            )
        steps = t.get("steps")
        if isinstance(steps, list):
            for group in steps:
                if not isinstance(group, list):
                    continue
                for step in group:
                    if isinstance(step, dict):
                        ref = step.get("template")
                        if isinstance(ref, str) and ref:
                            _edge(ref, container_id)


def _resolve_argo_references(
    per_file: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Resolve Argo workflow references into references edges (pass 2).

    Builds a (namespace, kind, name) -> node_id index from workflow-level
    argo:// nodes only (3 path segments). For each candidate that resolves,
    appends an EXTRACTED references edge; unresolved candidates get an
    AMBIGUOUS edge to a deduplicated placeholder node.
    """
    index = {}
    for node in all_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.startswith("argo://"):
            continue
        if "#" in node_id:
            continue
        rest = node_id[len("argo://"):]
        parts = rest.split("/")
        if len(parts) != 3:
            continue
        namespace, kind, name = parts
        index[(namespace, kind, name)] = node_id
    placeholder_ids = set()
    for entry in per_file:
        for candidate in entry.get("argo_candidates") or []:
            target = index.get(
                (
                    candidate["namespace"],
                    candidate["target_kind"],
                    candidate["target_name"],
                )
            )
            if target is None:
                unresolved_id = (
                    f"argo://{candidate['namespace']}/"
                    f"{candidate['target_kind']}/{candidate['target_name']}#unresolved"
                )
                all_edges.append(
                    {
                        "source": candidate["source"],
                        "target": unresolved_id,
                        "relation": "references",
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
                            "file_type": "argo",
                            "source_file": candidate["source_file"],
                            "attributes": {"unresolved": True},
                        }
                    )
                continue
            all_edges.append(
                {
                    "source": candidate["source"],
                    "target": target,
                    "relation": "references",
                    "confidence": "EXTRACTED",
                    "source_file": candidate["source_file"],
                }
            )


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
    # Selector->labels pass: a Service whose selector is a subset of a
    # same-namespace workload's labels selects that workload.
    services = []
    workloads = []
    for node in all_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.startswith("k8s://"):
            continue
        attributes = node.get("attributes") or {}
        kind = attributes.get("kind")
        if kind == "Service" and attributes.get("selector"):
            services.append(node)
        elif kind != "Service" and attributes.get("labels"):
            workloads.append(node)
    for svc in services:
        svc_id = svc["id"]
        svc_ns = svc_id[len("k8s://"):].split("/", 2)[0]
        selector = (svc.get("attributes") or {}).get("selector") or {}
        for workload in workloads:
            workload_id = workload["id"]
            workload_ns = workload_id[len("k8s://"):].split("/", 2)[0]
            if workload_ns != svc_ns:
                continue
            labels = (workload.get("attributes") or {}).get("labels") or {}
            if all(labels.get(k) == v for k, v in selector.items()):
                all_edges.append(
                    {
                        "source": svc_id,
                        "target": workload_id,
                        "relation": "selects",
                        "confidence": "INFERRED",
                        "source_file": svc["source_file"],
                    }
                )


def _extract_kustomize(path: Path, raw_text: str) -> dict:
    """Extract a kustomization node plus one generated-resource node per
    configMapGenerator/secretGenerator entry and a generates edge per node."""
    dirname = path.parent.as_posix()
    attributes = {"dir": dirname, "namespace": "_cluster"}
    nodes = []
    edges = []
    for doc in yaml.safe_load_all(raw_text):
        if not isinstance(doc, dict):
            continue
        namespace = doc.get("namespace")
        if isinstance(namespace, str) and namespace:
            attributes["namespace"] = namespace
        for key in ("resources", "bases", "components"):
            value = doc.get(key)
            if isinstance(value, list):
                attributes[key] = value
    node = {
        "id": f"kustomize://{dirname}/{path.name}",
        "label": path.name,
        "file_type": "kustomize",
        "source_file": str(path),
        "source_location": "doc0",
        "attributes": attributes,
    }
    nodes.append(node)
    for doc in yaml.safe_load_all(raw_text):
        if not isinstance(doc, dict):
            continue
        for generator, kind in (
            ("configMapGenerator", "ConfigMap"),
            ("secretGenerator", "Secret"),
        ):
            entries = doc.get(generator)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    continue
                generated_namespace = entry.get("namespace") or "_cluster"
                generated_id = f"k8s://{generated_namespace}/{kind}/{name}"
                nodes.append(
                    {
                        "id": generated_id,
                        "label": f"{kind}/{name}",
                        "file_type": "k8s",
                        "source_file": str(path),
                        "source_location": "doc0",
                        "attributes": {
                            "kind": kind,
                            "generated_by": node["id"],
                            "generator": generator,
                            "name": name,
                        },
                    }
                )
                edges.append(
                    {
                        "source": node["id"],
                        "target": generated_id,
                        "relation": "generates",
                        "confidence": "EXTRACTED",
                        "source_file": str(path),
                    }
                )
    return {"nodes": nodes, "edges": edges, "k8s_candidates": []}



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
