"""Kubernetes YAML extractor — dialect detection and node extraction."""
from __future__ import annotations

import posixpath
import shlex
from enum import Enum, auto
from pathlib import Path

import yaml

from graphify.extractors.images import image_ref_node

# Built-in k8s API groups that are NOT custom-resource groups. A resource whose
# apiVersion group is one of these (or ends in .k8s.io) is core infrastructure,
# never a CR that names a CRD.
_K8S_CORE_GROUPS = frozenset(
    {
        "apps",
        "batch",
        "autoscaling",
        "policy",
        "extensions",
        "networking",
        "storage",
        "scheduling",
        "admissionregistration",
        "apiextensions",
        "apiregistration",
        "authentication",
        "authorization",
        "certificates",
        "coordination",
        "discovery",
        "events",
        "flowcontrol",
        "node",
        "rbac",
    }
)


class Dialect(Enum):
    K8S_MANIFEST = auto()
    ARGO_WORKFLOW = auto()
    KUSTOMIZATION = auto()
    CI_WORKFLOW = auto()


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
    saw_ci = False
    all_ci = True
    for doc in docs:
        if not isinstance(doc, dict):
            return None
        if "jobs" in doc and "apiVersion" not in doc and "kind" not in doc:
            saw_ci = True
            all_argo = False
            all_kustomize = False
            continue
        all_ci = False
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
    if all_ci:
        return Dialect.CI_WORKFLOW
    if saw_argo or saw_kustomize or saw_ci:
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
    if dialect is Dialect.CI_WORKFLOW:
        return _extract_ci(path, raw_text)
    if dialect is not Dialect.K8S_MANIFEST:
        return {"nodes": [], "edges": [], "k8s_candidates": []}
    nodes = []
    edges = []
    candidates = []
    seen_image_ids: set = set()
    seen_edges: set = set()
    for i, doc in enumerate(yaml.safe_load_all(raw_text)):
        metadata = doc.get("metadata") or {}
        kind = doc.get("kind", "")
        name = metadata.get("name", "")
        namespace = metadata.get("namespace") or "_cluster"
        attributes = {"kind": kind, "namespace": namespace}
        # CRD group+kind and CR apiVersion group power the CRD→CR `defines` link
        # (R7-S2): a CR's apiVersion group matches the CRD's spec.group, and its
        # kind matches the CRD's spec.names.kind.
        api_version = doc.get("apiVersion")
        doc_spec = doc.get("spec")
        if kind == "CustomResourceDefinition" and isinstance(doc_spec, dict):
            if isinstance(doc_spec.get("group"), str):
                attributes["crd_group"] = doc_spec["group"]
            names = doc_spec.get("names")
            if isinstance(names, dict) and isinstance(names.get("kind"), str):
                attributes["crd_kind"] = names["kind"]
        elif isinstance(api_version, str) and "/" in api_version:
            group = api_version.split("/", 1)[0]
            # Stamp only CUSTOM resources (CRs). Built-in groups (apps, batch,
            # *.k8s.io, autoscaling, policy, extensions) are not CRs and never
            # name a CRD, so leaving them unstamped avoids attribute churn.
            if group and not group.endswith(".k8s.io") and group not in _K8S_CORE_GROUPS:
                attributes["api_group"] = group
        # cert-manager Certificate references its Issuer/ClusterIssuer by
        # spec.issuerRef, and its Service by spec.dnsNames (`<svc>.<ns>.svc…`)
        # (R7-S4).
        if kind == "Certificate" and isinstance(doc.get("spec"), dict):
            issuer_ref = (doc_spec or {}).get("issuerRef")
            if isinstance(issuer_ref, dict):
                if isinstance(issuer_ref.get("kind"), str):
                    attributes["issuer_kind"] = issuer_ref["kind"]
                if isinstance(issuer_ref.get("name"), str):
                    attributes["issuer_name"] = issuer_ref["name"]
            dns_names = (doc_spec or {}).get("dnsNames")
            if isinstance(dns_names, list):
                attributes["cert_dns_names"] = [
                    d for d in dns_names if isinstance(d, str)
                ]
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
        workload_id = f"k8s://{namespace}/{kind}/{name}"
        nodes.append(
            {
                "id": workload_id,
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
        container_list = None
        spec = doc.get("spec")
        if isinstance(spec, dict):
            template = spec.get("template")
            if isinstance(template, dict) and isinstance(template.get("spec"), dict):
                container_list = template["spec"].get("containers")
            elif isinstance(spec.get("containers"), list):
                container_list = spec["containers"]
        if isinstance(container_list, list):
            for container in container_list:
                if not isinstance(container, dict):
                    continue
                image = container.get("image")
                if not isinstance(image, str):
                    continue
                img_node = image_ref_node(image)
                if img_node is None:
                    continue
                if img_node["id"] not in seen_image_ids:
                    seen_image_ids.add(img_node["id"])
                    nodes.append(dict(img_node))
                if (workload_id, img_node["id"]) not in seen_edges:
                    seen_edges.add((workload_id, img_node["id"]))
                    edges.append(
                        {
                            "source": workload_id,
                            "target": img_node["id"],
                            "relation": "runs",
                            "confidence": "EXTRACTED",
                            "source_file": str(path),
                        }
                    )
        if kind == "Ingress":
            for svc_name in _ingress_service_names(doc):
                svc_id = f"k8s://{namespace}/Service/{svc_name}"
                if (workload_id, svc_id) not in seen_edges:
                    seen_edges.add((workload_id, svc_id))
                    edges.append(
                        {
                            "source": workload_id,
                            "target": svc_id,
                            "relation": "routes",
                            "confidence": "EXTRACTED",
                            "source_file": str(path),
                        }
                    )
    return {"nodes": nodes, "edges": edges, "k8s_candidates": candidates}


def _ingress_service_names(doc: dict) -> list[str]:
    """Service names a k8s Ingress routes to, from every rule path backend and
    the default backend. Non-service backends (resource refs) are ignored, and
    names are deduped (a service referenced by several paths/hosts appears once)."""
    names: list[str] = []
    seen: set[str] = set()
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return names

    def _add(backend: object) -> None:
        if not isinstance(backend, dict):
            return
        svc = backend.get("service")
        if isinstance(svc, dict) and isinstance(svc.get("name"), str):
            if svc["name"] not in seen:
                seen.add(svc["name"])
                names.append(svc["name"])

    default = spec.get("defaultBackend")
    if isinstance(default, dict):
        _add(default)
    rules = spec.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            http = rule.get("http")
            if not isinstance(http, dict):
                continue
            paths = http.get("paths")
            if not isinstance(paths, list):
                continue
            for path in paths:
                if isinstance(path, dict):
                    _add(path.get("backend"))
    return names


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
    # CRD→CR `defines` pass: a CustomResource instance's apiVersion group + kind
    # names the CRD that defines its schema (R7-S2). Index CRDs by (group, kind)
    # and link every matching CR. Exact match, so EXTRACTED; no node fabricated —
    # a CR whose CRD is absent (lives in another repo/file) is simply unlinked.
    crds_by_kind: dict[tuple[str, str], dict] = {}
    crs: list[dict] = []
    for node in all_nodes:
        attributes = node.get("attributes") or {}
        if attributes.get("crd_group") and attributes.get("crd_kind"):
            crds_by_kind[(attributes["crd_group"], attributes["crd_kind"])] = node
        elif attributes.get("api_group") and attributes.get("kind"):
            crs.append(node)
    seen_defines: set[tuple[str, str]] = set()
    for cr in crs:
        attributes = cr.get("attributes") or {}
        crd = crds_by_kind.get((attributes["api_group"], attributes["kind"]))
        if crd is None or (crd["id"], cr["id"]) in seen_defines:
            continue
        seen_defines.add((crd["id"], cr["id"]))
        all_edges.append(
            {
                "source": crd["id"],
                "target": cr["id"],
                "relation": "defines",
                "confidence": "EXTRACTED",
                "source_file": crd["source_file"],
            }
        )
    # TLS joins (R7-S4): a Certificate's issuerRef names its Issuer/ClusterIssuer,
    # and its dnsNames (`<svc>.<ns>.svc[.cluster.local]`) name the Service it
    # serves. Exact matches, EXTRACTED; an absent issuer/service is left unlinked.
    certs = [n for n in all_nodes if (n.get("attributes") or {}).get("kind") == "Certificate"]
    if certs:
        # Issuer/ClusterIssuer index: id `k8s://<ns>/<Kind>/<name>`.
        issuer_index: dict[tuple[str, str], dict] = {}
        for node in all_nodes:
            attrs = node.get("attributes") or {}
            if attrs.get("kind") in ("Issuer", "ClusterIssuer"):
                node_id = node.get("id", "")
                _, kind, name = node_id[len("k8s://"):].split("/", 2)
                issuer_index[(kind, name)] = node
        service_index: dict[tuple[str, str], dict] = {}
        for node in all_nodes:
            attrs = node.get("attributes") or {}
            if attrs.get("kind") == "Service":
                node_id = node.get("id", "")
                namespace, _, name = node_id[len("k8s://"):].split("/", 2)
                service_index[(namespace, name)] = node
        seen_issues: set[tuple[str, str]] = set()
        seen_serves: set[tuple[str, str]] = set()
        for cert in certs:
            attrs = cert.get("attributes") or {}
            issuer_kind = attrs.get("issuer_kind")
            issuer_name = attrs.get("issuer_name")
            if issuer_kind and issuer_name:
                issuer = issuer_index.get((issuer_kind, issuer_name))
                if issuer is not None and (issuer["id"], cert["id"]) not in seen_issues:
                    seen_issues.add((issuer["id"], cert["id"]))
                    all_edges.append(
                        {
                            "source": issuer["id"],
                            "target": cert["id"],
                            "relation": "issues",
                            "confidence": "EXTRACTED",
                            "source_file": issuer["source_file"],
                        }
                    )
            for dns in attrs.get("cert_dns_names") or []:
                # `<svc>.<namespace>.svc[.cluster.local]` → (namespace, svc).
                parts = dns.split(".")
                if len(parts) < 2 or "svc" not in parts:
                    continue
                svc_name = parts[0]
                namespace = parts[1]
                svc = service_index.get((namespace, svc_name))
                if svc is not None and (cert["id"], svc["id"]) not in seen_serves:
                    seen_serves.add((cert["id"], svc["id"]))
                    all_edges.append(
                        {
                            "source": cert["id"],
                            "target": svc["id"],
                            "relation": "serves",
                            "confidence": "EXTRACTED",
                            "source_file": cert["source_file"],
                        }
                    )


def _resolve_kustomize_includes(
    per_file: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Resolve Kustomize includes candidates into EXTRACTED edges (pass 2).

    Indexes Kustomize/K8s nodes by the (directory, basename) of their
    source_file, then appends an includes edge for each candidate whose
    resolved path matches an indexed node.
    """
    index = {}
    for node in all_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        if not (node_id.startswith("kustomize://") or node_id.startswith("k8s://")):
            continue
        if "#" in node_id:
            continue
        attributes = node.get("attributes") or {}
        if "generator" in attributes:
            continue
        source_file = node.get("source_file")
        if not isinstance(source_file, str):
            continue
        source_path = Path(source_file)
        index[(source_path.parent, source_path.name)] = node_id
    placeholder_ids = set()
    for entry in per_file:
        for candidate in entry.get("kustomize_candidates") or []:
            resolved = posixpath.normpath(
                posixpath.join(candidate["dir"], candidate["target_path"])
            )
            target = index.get((Path(resolved).parent, Path(resolved).name))
            if target is None:
                unresolved_id = f"kustomize://{resolved}#unresolved"
                all_edges.append(
                    {
                        "source": candidate["source"],
                        "target": unresolved_id,
                        "relation": "includes",
                        "confidence": "AMBIGUOUS",
                        "source_file": candidate["source_file"],
                    }
                )
                if unresolved_id not in placeholder_ids:
                    placeholder_ids.add(unresolved_id)
                    all_nodes.append(
                        {
                            "id": unresolved_id,
                            "label": f"{Path(resolved).name} (unresolved)",
                            "file_type": "kustomize",
                            "source_file": candidate["source_file"],
                            "attributes": {"unresolved": True},
                        }
                    )
                continue
            all_edges.append(
                {
                    "source": candidate["source"],
                    "target": target,
                    "relation": "includes",
                    "confidence": "EXTRACTED",
                    "source_file": candidate["source_file"],
                }
            )


def _extract_kustomize(path: Path, raw_text: str) -> dict:
    """Extract a kustomization node plus one generated-resource node per
    configMapGenerator/secretGenerator entry and a generates edge per node."""
    dirname = path.parent.as_posix()
    kustomize_id = f"kustomize://{dirname}/{path.name}"
    attributes = {"dir": dirname, "namespace": "_cluster"}
    candidates = []
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
                for entry in value:
                    if isinstance(entry, str) and not (
                        "://" in entry
                        or entry.startswith("http")
                        or entry.startswith("git")
                    ):
                        candidates.append(
                            {
                                "source": kustomize_id,
                                "target_path": entry,
                                "dir": dirname,
                                "source_file": str(path),
                                "relation": "includes",
                            }
                        )
    node = {
        "id": kustomize_id,
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
    return {
        "nodes": nodes,
        "edges": edges,
        "k8s_candidates": [],
        "kustomize_candidates": candidates,
    }



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


def _extract_ci(path: Path, raw_text: str) -> dict:
    """Extract one node per top-level jobs.<key> in a CI workflow document.

    Docker build/push steps also emit image nodes and publishes edges from
    the job node to each image the step produces; every step with a uses:
    ref additionally emits an action node and a references edge to it.
    """
    nodes = []
    edges = []
    seen_image_ids: set = set()
    seen_action_refs: set = set()
    seen_edges: set = set()
    dirname = path.parent.as_posix()
    for doc in yaml.safe_load_all(raw_text):
        if not isinstance(doc, dict):
            continue
        jobs = doc.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_key, job_value in jobs.items():
            if not isinstance(job_key, str):
                continue
            if not isinstance(job_value, dict):
                continue
            job_id = f"ci://{dirname}/{job_key}"
            nodes.append(
                {
                    "id": job_id,
                    "label": job_key,
                    "file_type": "ci",
                    "source_file": str(path),
                    "source_location": "doc0",
                }
            )
            steps = job_value.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                candidates = []
                uses = step.get("uses")
                run = step.get("run")
                if isinstance(uses, str) and uses:
                    action_id = f"ci://{dirname}/_action/{uses}"
                    if uses not in seen_action_refs:
                        seen_action_refs.add(uses)
                        nodes.append(
                            {
                                "id": action_id,
                                "label": uses,
                                "file_type": "ci",
                                "source_file": str(path),
                                "source_location": "doc0",
                            }
                        )
                    if (job_id, action_id) not in seen_edges:
                        seen_edges.add((job_id, action_id))
                        edges.append(
                            {
                                "source": job_id,
                                "target": action_id,
                                "relation": "references",
                                "confidence": "EXTRACTED",
                                "source_file": str(path),
                            }
                        )
                if isinstance(uses, str) and uses.startswith(
                    "docker/build-push-action"
                ):
                    with_ = step.get("with")
                    if isinstance(with_, dict):
                        tags = with_.get("tags")
                        if isinstance(tags, str):
                            candidates.append(tags)
                elif isinstance(run, str) and any(
                    marker in run
                    for marker in ("docker build", "docker push", "build-and-push")
                ):
                    tokens = shlex.split(run)
                    if "docker push" in run:
                        for token in tokens:
                            if token not in ("docker", "push") and not token.startswith(
                                "-"
                            ):
                                candidates.append(token)
                                break
                    else:
                        for i, token in enumerate(tokens):
                            if token in ("-t", "--tag") and i + 1 < len(tokens):
                                image_arg = tokens[i + 1]
                                if "${{" in image_arg or "}}" in image_arg:
                                    break
                                candidates.append(image_arg)
                                break
                for candidate in candidates:
                    img_node = image_ref_node(candidate)
                    if img_node is None:
                        continue
                    if img_node["id"] not in seen_image_ids:
                        seen_image_ids.add(img_node["id"])
                        nodes.append(dict(img_node))
                    if (job_id, img_node["id"]) not in seen_edges:
                        seen_edges.add((job_id, img_node["id"]))
                        edges.append(
                            {
                                "source": job_id,
                                "target": img_node["id"],
                                "relation": "publishes",
                                "confidence": "EXTRACTED",
                                "source_file": str(path),
                            }
                        )
    return {"nodes": nodes, "edges": edges, "k8s_candidates": []}
