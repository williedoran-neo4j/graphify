"""graphdb — moved verbatim from graphify/export.py."""
from __future__ import annotations

import json

from graphify.analyze import _node_community_map
from graphify.embed import _embed_space
from graphify.search import load_sidecar
import networkx as nx
import os
import re


def _embedding_index(embeddings_path):
    """Map sidecar id -> {space: list of floats}, or None when no sidecar
    was given or the given path holds no stored vectors.

    Each node id lives in exactly one sidecar group: ``text_ids`` /
    ``text_vecs`` (the "text" space) or ``code_ids`` / ``code_vecs`` (the
    "code" space), so an id maps only to its own group's row. A sidecar with
    no code group (pre-R8) falls back to the text row for every id, as the
    exporter always did.
    """
    if embeddings_path is None:
        return None
    sidecar = load_sidecar(embeddings_path)
    if sidecar is None:
        return None
    import numpy as np

    text_ids = sidecar["text_ids"]
    text_vecs = np.asarray(sidecar["text_vecs"])
    include = {
        str(nid): {"text": row.tolist()}
        for nid, row in zip(text_ids, text_vecs, strict=False)
    }
    code_ids = sidecar.get("code_ids")
    code_vecs = sidecar.get("code_vecs")
    if code_ids is not None and code_vecs is not None:
        code_vecs = np.asarray(code_vecs)
        for nid, row in zip(code_ids, code_vecs, strict=False):
            include[str(nid)]["code"] = row.tolist()
    else:
        for entry in include.values():
            entry["code"] = entry["text"]
    if not include:
        return None
    return include


def _sidecar_dim(embeddings_path):
    """Return each space's sidecar ``meta["dim"]``: ``{"text": dim,
    "code": dim}``, or None when no sidecar."""
    if embeddings_path is None:
        return None
    sidecar = load_sidecar(embeddings_path)
    if sidecar is None:
        return None
    text_dim = int(json.loads(str(sidecar["text_meta"]))["dim"])
    # A pre-code sidecar has no code group; the code space reuses the text
    # dim (and, in _embedding_index, the text row) exactly as it always did.
    dims = {"text": text_dim, "code": text_dim}
    code_meta = sidecar.get("code_meta")
    if code_meta is not None:
        dims["code"] = int(json.loads(str(code_meta))["dim"])
    return dims


def _vector_index_ddl(space: str, dim: int) -> str:
    """Idempotent vector-index DDL for one embedding space, dimensioned by ``dim``."""
    return (
        f"CREATE VECTOR INDEX graphify_{space}_embeddings IF NOT EXISTS "
        f"FOR (n:Embedded) ON (n.embedding_{space}) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


def _emit_vector_indexes(run, dims: dict) -> None:
    """Emit one vector index per space, each dimensioned by its own sidecar
    group's dim; a driver that cannot build the index must not fail the push."""
    for space, dim in dims.items():
        try:
            result = run(_vector_index_ddl(space, dim))
            if result is not None and hasattr(result, "consume"):
                result.consume()
        except Exception:
            pass


def _embedding_prop(embedding_index, node_id, file_type, props):
    """Splice the sidecar's vector for ``node_id`` (when present) into ``props``
    under ``embedding_<space>``, keyed by the node's file_type's space."""
    if embedding_index is None:
        return props
    space = _embed_space(file_type)
    if space is None:
        return props
    row = embedding_index.get(node_id)
    if row is None:
        return props
    props[f"embedding_{space}"] = row[space]
    return props


def _pushable_props(data: dict) -> dict:
    """Return the properties from ``data`` that can be pushed to a graph DB.

    Keeps scalar values (str/int/float/bool) and lists whose every member is a
    number (int/float, never bool). A list with a bool member or any nested/
    object member (e.g. dict) is dropped whole.
    """
    props = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool)):
            props[k] = v
        elif isinstance(v, list) and all(
            isinstance(m, (int, float)) and not isinstance(m, bool) for m in v
        ):
            props[k] = v
    return props


def push_to_neo4j(
    G: nx.Graph,
    uri: str,
    user: str,
    password: str,
    communities: dict[int, list[str]] | None = None,
    embeddings_path: str | os.PathLike | None = None,
) -> dict[str, int]:
    """Push graph directly to a running Neo4j instance via the Python driver.

    Requires: pip install neo4j

    Uses MERGE so re-running is safe - nodes and edges are upserted, not duplicated.
    Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j driver not installed. Run: pip install neo4j"
        ) from e

    node_community = _node_community_map(communities) if communities else {}

    def _safe_rel(relation: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"

    def _safe_label(label: str) -> str:
        """Sanitize a Neo4j node label to prevent Cypher injection."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
        return sanitized if sanitized else "Entity"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    embedding_index = _embedding_index(embeddings_path)
    dims = _sidecar_dim(embeddings_path)
    nodes_pushed = 0
    edges_pushed = 0

    with driver.session() as session:
        for node_id, data in G.nodes(data=True):
            props = _pushable_props(data)
            props["id"] = node_id
            _embedding_prop(embedding_index, node_id, data.get("file_type"), props)
            embedded = "embedding_text" in props or "embedding_code" in props
            cid = node_community.get(node_id)
            if cid is not None:
                props["community"] = cid
            ftype = _safe_label(data.get("file_type", "Entity").capitalize())
            session.run(
                f"MERGE (n:{ftype} {{id: $id}})"
                f"{' SET n:Embedded' if embedded else ''} SET n += $props",
                id=node_id,
                props=props,
            )
            nodes_pushed += 1

        if dims is not None:
            _emit_vector_indexes(session.run, dims)

        for u, v, data in G.edges(data=True):
            rel = _safe_rel(data.get("relation", "RELATED_TO"))
            props = _pushable_props(data)
            session.run(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += $props",
                src=u,
                tgt=v,
                props=props,
            )
            edges_pushed += 1

    driver.close()
    return {"nodes": nodes_pushed, "edges": edges_pushed}

def push_to_falkordb(
    G: nx.Graph,
    uri: str,
    user: str | None = None,
    password: str | None = None,
    communities: dict[int, list[str]] | None = None,
    graph_name: str = "graphify",
    embeddings_path: str | os.PathLike | None = None,
) -> dict[str, int]:
    """Push graph directly to a running FalkorDB instance via the Python SDK.

    Requires: pip install falkordb

    FalkorDB is OpenCypher-compatible, so the MERGE/SET upsert queries are
    identical to push_to_neo4j. Differences from the Neo4j path:
      - connects with FalkorDB(host, port, username, password) instead of a bolt
        driver; only the host/port are read from the URI, so the scheme is
        informational - "falkordb://localhost:6379", "redis://localhost:6379"
        and a bare "localhost:6379" are all equivalent (default port 6379).
      - a named graph is selected via db.select_graph(graph_name) (default
        "graphify"); FalkorDB keys each graph by name in the same instance.
      - queries run via graph.query(cypher, params) - there is no session object.
      - auth is optional (FalkorDB runs without credentials by default), so user
        and password may be None.
      - no APOC: the Neo4j path does not use APOC either, so nothing to port.

    Uses MERGE so re-running is safe - nodes and edges are upserted, not
    duplicated. Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from falkordb import FalkorDB
    except ImportError as e:
        raise ImportError(
            "falkordb SDK not installed. Run: pip install falkordb"
        ) from e

    from urllib.parse import urlparse

    node_community = _node_community_map(communities) if communities else {}

    def _safe_rel(relation: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"

    def _safe_label(label: str) -> str:
        """Sanitize a FalkorDB node label to prevent Cypher injection."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
        return sanitized if sanitized else "Entity"

    parsed = urlparse(uri if "://" in uri else f"redis://{uri}")
    # FalkorDB auth is optional. Only send credentials when a password is
    # provided; otherwise connect anonymously and ignore any bolt-style default
    # username (e.g. Neo4j's "neo4j"), which FalkorDB rejects as an unknown ACL
    # user. Credentials embedded in the URI take precedence over the args.
    connect_user = parsed.username or (user if password else None)
    connect_password = parsed.password or (password or None)
    db = FalkorDB(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        username=connect_user,
        password=connect_password,
    )
    graph = db.select_graph(graph_name)
    embedding_index = _embedding_index(embeddings_path)
    dims = _sidecar_dim(embeddings_path)
    nodes_pushed = 0
    edges_pushed = 0

    for node_id, data in G.nodes(data=True):
        props = _pushable_props(data)
        props["id"] = node_id
        _embedding_prop(embedding_index, node_id, data.get("file_type"), props)
        embedded = "embedding_text" in props or "embedding_code" in props
        cid = node_community.get(node_id)
        if cid is not None:
            props["community"] = cid
        ftype = _safe_label(data.get("file_type", "Entity").capitalize())
        graph.query(
            f"MERGE (n:{ftype} {{id: $id}})"
            f"{' SET n:Embedded' if embedded else ''} SET n += $props",
            {"id": node_id, "props": props},
        )
        nodes_pushed += 1

    if dims is not None:
        _emit_vector_indexes(graph.query, dims)

    for u, v, data in G.edges(data=True):
        rel = _safe_rel(data.get("relation", "RELATED_TO"))
        props = _pushable_props(data)
        graph.query(
            f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += $props",
            {"src": u, "tgt": v, "props": props},
        )
        edges_pushed += 1

    return {"nodes": nodes_pushed, "edges": edges_pushed}
