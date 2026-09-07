"""`merge-graphs` links a package dependency to its defining package across repos (R6).

A package manifest emits a canonical ``pkg_<name>`` node plus ``depends_on`` edges.
When a dependency's own manifest is in another repo, the merged graph holds that
dependency in two prefixed forms — the external reference minted in the dependent's
repo and the definition in the owning repo — and a traversal cannot cross. The merge
now adds a cross-repo ``depends_on`` edge between them, mirroring ``same_type_as``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _run(args, cwd):
    return subprocess.run([PYTHON, "-m", "graphify"] + args, cwd=cwd,
                          capture_output=True, text=True)


def _pkg_node(nid: str, label: str, source_file: str | None):
    node: dict = {
        "id": nid,
        "label": label,
        "file_type": "code",
        "type": "package",
        "ecosystem": "python",
    }
    if source_file is not None:
        node["source_file"] = source_file
    return node


def _write(p: Path, nodes: list[dict], links: list[dict] | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "directed": True, "multigraph": False, "graph": {},
        "nodes": nodes, "links": links or [],
    }))


def _merge(tmp_path, left, right):
    a = tmp_path / "svc_a" / "graphify-out" / "graph.json"
    b = tmp_path / "svc_b" / "graphify-out" / "graph.json"
    _write(a, left["nodes"], left.get("links"))
    _write(b, right["nodes"], right.get("links"))
    out = tmp_path / "merged.json"
    r = _run(["merge-graphs", str(a), str(b), "--out", str(out)], tmp_path)
    assert r.returncode == 0, f"merge failed: {r.stderr}"
    data = json.loads(out.read_text())
    nodes = data["nodes"]
    links = [e for e in data["links"] if e.get("relation") == "depends_on"]
    return nodes, links, data


def test_cross_repo_package_dependency_is_linked(tmp_path):
    # svc_a's package depends on a package defined in svc_b.
    left = {
        "nodes": [
            _pkg_node("pkg_consumer", "consumer", "consumer/pyproject.toml"),
            # external reference minted by C1: same id, source_file=None
            _pkg_node("pkg_shared_lib", "shared-lib", None),
        ],
        "links": [
            {"source": "pkg_consumer", "target": "pkg_shared_lib",
             "relation": "depends_on", "confidence": "EXTRACTED",
             "context": "dependency"},
        ],
    }
    right = {
        "nodes": [_pkg_node("pkg_shared_lib", "shared-lib", "shared/pyproject.toml")],
    }
    nodes, links, _ = _merge(tmp_path, left, right)

    # Both repos keep their own copy of the package node.
    ids = {n["id"] for n in nodes}
    assert "svc_a::pkg_shared_lib" in ids
    assert "svc_b::pkg_shared_lib" in ids

    # Exactly one cross-repo depends_on link is added, with prefixed endpoints.
    cross = [l for l in links if l.get("context") == "cross_repo"]
    assert len(cross) == 1, cross
    endpoints = {cross[0]["source"], cross[0]["target"]}
    assert endpoints == {"svc_a::pkg_consumer", "svc_b::pkg_shared_lib"}
    assert cross[0]["confidence"] == "INFERRED"


def test_same_repo_dependency_is_not_linked(tmp_path):
    # Both nodes live in the same repo: no cross-repo link.
    left = {
        "nodes": [
            _pkg_node("pkg_consumer", "consumer", "consumer/pyproject.toml"),
            _pkg_node("pkg_shared_lib", "shared-lib", "shared/pyproject.toml"),
        ],
        "links": [
            {"source": "pkg_consumer", "target": "pkg_shared_lib",
             "relation": "depends_on", "confidence": "EXTRACTED"},
        ],
    }
    right = {"nodes": [_pkg_node("pkg_unrelated", "unrelated", "u/pyproject.toml")]}
    _, links, _ = _merge(tmp_path, left, right)
    assert [l for l in links if l.get("context") == "cross_repo"] == []


def test_dependency_with_no_cross_repo_definition_is_not_linked(tmp_path):
    # The dependency is external to BOTH repos: reference node only, no definition.
    left = {
        "nodes": [
            _pkg_node("pkg_consumer", "consumer", "consumer/pyproject.toml"),
            _pkg_node("pkg_django", "django", None),
        ],
        "links": [
            {"source": "pkg_consumer", "target": "pkg_django",
             "relation": "depends_on", "confidence": "EXTRACTED"},
        ],
    }
    right = {"nodes": [_pkg_node("pkg_unrelated", "unrelated", "u/pyproject.toml")]}
    _, links, _ = _merge(tmp_path, left, right)
    assert [l for l in links if l.get("context") == "cross_repo"] == []


def test_reference_nodes_are_not_linked_as_definitions(tmp_path):
    # Both repos depend on the SAME external package (numpy): each holds only a
    # source_file=None reference, neither defines it. The link pass must NOT link
    # one repo's reference to the other's — there is no definition to join on.
    left = {
        "nodes": [
            _pkg_node("pkg_consumer_a", "consumer-a", "a/pyproject.toml"),
            _pkg_node("pkg_numpy", "numpy", None),  # external reference
        ],
        "links": [
            {"source": "pkg_consumer_a", "target": "pkg_numpy",
             "relation": "depends_on", "confidence": "EXTRACTED"},
        ],
    }
    right = {
        "nodes": [
            _pkg_node("pkg_consumer_b", "consumer-b", "b/pyproject.toml"),
            _pkg_node("pkg_numpy", "numpy", None),  # also just an external reference
        ],
        "links": [
            {"source": "pkg_consumer_b", "target": "pkg_numpy",
             "relation": "depends_on", "confidence": "EXTRACTED"},
        ],
    }
    _, links, _ = _merge(tmp_path, left, right)
    cross = [l for l in links if l.get("context") == "cross_repo"]
    assert cross == [], f"reference-to-reference links must not be created: {cross}"
