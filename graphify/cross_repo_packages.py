"""Join package dependencies across repositories in a merged graph (R6).

A package manifest (pyproject.toml / go.mod / Cargo.toml / pom.xml / apm.yml)
emits a canonical ``pkg_<name>`` node for the package it defines plus a
``depends_on`` edge to each dependency's canonical id. When a dependency's own
manifest lives in a *different* repo, that dependency arrives in the merged
graph in two forms: an external reference node (``source_file=None``) minted in
the depending repo via build_from_json (R6 C1), and the real definition node in
the repo that owns the package. After repo-prefixing neither is connected — both
are ``<repo>::pkg_<name>`` — so a traversal cannot cross from the dependent to
the definition.

This pass adds a cross-repo ``depends_on`` edge from the depending package node
to the foreign package node for every such shared id. Edges only, no node
merging: each repo keeps its own node and provenance, exactly like the
``same_type_as`` pass in :mod:`graphify.cross_repo_types`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx


def link_shared_package_dependencies(merged: "nx.Graph") -> int:
    """Link dependencies to their defining package across repos. Returns the
    count of cross-repo ``depends_on`` edges added.

    The join key is the bare canonical package id (``pkg_<name>``, repo prefix
    stripped). For every ``depends_on`` edge whose target is a package id, the
    pass finds every ``type=="package"`` node in a *different* repo carrying the
    same bare id and links the depending package node to it. Same-repo links and
    targets with no cross-repo definition are left untouched.
    """
    # bare package id -> node ids (every repo-prefixed node claiming that id).
    by_bare: dict[str, list[str]] = defaultdict(list)
    for node, data in merged.nodes(data=True):
        if data.get("type") == "package":
            by_bare[node.split("::", 1)[-1]].append(node)

    links: list[tuple[str, str]] = []
    for u, v, data in merged.edges(data=True):
        if data.get("relation") != "depends_on":
            continue
        src = data.get("_src", u)
        tgt = data.get("_tgt", v)
        src_data = merged.nodes.get(src)
        if not src_data or src_data.get("type") != "package":
            continue
        src_repo = src_data.get("repo")
        tgt_bare = tgt.split("::", 1)[-1]
        for foreign in by_bare.get(tgt_bare, ()):
            if foreign == tgt or merged.nodes.get(foreign, {}).get("repo") == src_repo:
                continue
            if merged.has_edge(src, foreign):
                continue
            links.append((src, foreign))

    for src, foreign in links:
        merged.add_edge(
            src,
            foreign,
            relation="depends_on",
            context="cross_repo",
            confidence="INFERRED",
            confidence_score=0.9,
            source_file=str(merged.nodes[src].get("source_file") or ""),
            weight=1.0,
            _src=src,
            _tgt=foreign,
        )
    return len(links)
