"""Env-file endpoint extractor (R7-S3): frontend → backend references.

Committed ``.env.*`` templates (``.env.preview``/``.env.trunk``/``.env.example``/
``.env.development``/``.env.post-build.*`` …) pin a frontend's runtime endpoints —
``VITE_OIDC_AUTHORITY=https://login.neo4j.com``, ``VITE_AURA_CONSOLE_API_URL=…``,
``GEN_AI_BASE_URL=…``. A bare ``.env`` / ``.env.local`` carries live secrets and is
deliberately NOT extracted (detect.py already drops it as a secret store).

Each ``http(s)://``-valued key becomes one canonical endpoint reference node keyed
by host (scheme + authority, case-folded), plus a ``points_to`` edge from the file's
own node, so a frontend's declared backends are queryable in the graph and can later
join onto a matching k8s Ingress host at merge time.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from graphify.extractors.base import _file_stem, _make_id

# Committed .env templates only — bare `.env` / `.env.local` are live secrets.
_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")
_URL_VALUE_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_env_endpoint_file(path: Path) -> bool:
    """True for a committed ``.env.*`` variant (not a live secret ``.env``)."""
    name = path.name.casefold()
    if not name.startswith(".env"):
        return False
    # Bare `.env` and `.env.local` (and `.env.*.local`) are live-secret files;
    # any other `.env.<suffix>` is a committed template/preview variant.
    return name != ".env" and not name.endswith(".local")


def _endpoint_id(host: str) -> str:
    """Canonical id for a host: ``endpoint://<lowercased-netloc>``."""
    return f"endpoint://{host.casefold()}"


def extract_env_endpoints(path: Path) -> dict:
    """Emit one file node + one endpoint node per ``http(s)://``-valued key."""
    nodes: list[dict] = []
    edges: list[dict] = []
    file_id = f"{_file_stem(path)}" or _make_id("env", path.name)
    file_id = _make_id("env", file_id) if not file_id.startswith("env_") else file_id

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"nodes": [], "edges": [], "k8s_candidates": []}

    file_node = {
        "id": file_id,
        "label": path.name,
        "file_type": "concept",
        "source_file": str(path),
        "attributes": {},
    }
    nodes.append(file_node)

    seen: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if not _URL_VALUE_RE.match(value):
            continue
        try:
            parsed = urlsplit(value)
        except ValueError:
            continue
        host = parsed.netloc
        # A DSN-style URL (`https://<key>@<host>/…`) carries userinfo; strip it so
        # the endpoint keys on the real authority, not the secret.
        if "@" in host:
            host = host.rsplit("@", 1)[1]
        # Skip a bare hostname placeholder (no scheme authority).
        if not host:
            continue
        eid = _endpoint_id(host)
        scheme = parsed.scheme or "https"
        if eid not in seen:
            seen.add(eid)
            nodes.append(
                {
                    "id": eid,
                    "label": f"{scheme}://{host}",
                    "file_type": "concept",
                    "source_file": str(path),
                    "attributes": {"endpoint_key": key.strip()},
                }
            )
        if (file_id, eid) not in seen_edges:
            seen_edges.add((file_id, eid))
            edges.append(
                {
                    "source": file_id,
                    "target": eid,
                    "relation": "points_to",
                    "confidence": "EXTRACTED",
                    "source_file": str(path),
                }
            )
    return {"nodes": nodes, "edges": edges, "k8s_candidates": []}
