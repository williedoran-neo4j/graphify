"""Committed .env* files expose frontend → backend endpoint references (R7-S3).

runtime config (VITE_* / *_URL / *_AUTHORITY / *_AUDIENCE) with http(s)://
values become deterministic endpoint reference nodes + `points_to` edges from the
file's own node, so a frontend's declared backends are queryable in the graph.
"""
from __future__ import annotations

from graphify.extractors.env_endpoints import extract_env_endpoints, is_env_endpoint_file


def test_is_env_endpoint_file_matches_committed_env_variants():
    from pathlib import Path
    assert is_env_endpoint_file(Path(".env.preview"))
    assert is_env_endpoint_file(Path(".env.trunk"))
    assert is_env_endpoint_file(Path(".env.example"))
    assert is_env_endpoint_file(Path(".env.development"))
    assert is_env_endpoint_file(Path("apps/workspace/.env.post-build.example"))
    # Bare .env / .env.local (live secrets) are NOT extracted.
    assert not is_env_endpoint_file(Path(".env"))
    assert not is_env_endpoint_file(Path(".env.local"))


def test_extract_env_endpoints_emits_host_reference_nodes(tmp_path):
    """A .env file with VITE_*/URL keys yields one file node + one endpoint
    reference node per http(s):// value, with a `points_to` edge to each."""
    env = tmp_path / ".env.preview"
    env.write_text(
        """# frontend runtime config
VITE_OIDC_AUTHORITY=https://login.neo4j.com
VITE_AURA_CONSOLE_API_URL=https://console.neo4j.io
VITE_OIDC_AUDIENCE=https://console.neo4j.io
VITE_SEGMENT_API_KEY=UmmjVrTeoCnRWD (not a URL)
VITE_QA_HOSTNAME=localhost
""",
        encoding="utf-8",
    )

    result = extract_env_endpoints(env)
    nodes = {n["id"]: n for n in result["nodes"]}
    edges = result["edges"]

    # Endpoint nodes keyed by host (scheme + authority), deterministic.
    console = "endpoint://console.neo4j.io"
    login = "endpoint://login.neo4j.com"
    assert console in nodes
    assert login in nodes
    assert nodes[console]["label"] == "https://console.neo4j.io"
    assert nodes[login]["label"] == "https://login.neo4j.com"
    for nid in (console, login):
        assert nodes[nid]["file_type"] == "concept"

    # Both VITE_AURA_CONSOLE_API_URL and VITE_OIDC_AUDIENCE point at console.neo4j.io
    # → one node, two points_to edges from the file node (deduped by endpoint id).
    file_id = next(n for n in nodes if n != console and n != login)
    points = [e for e in edges if e.get("relation") == "points_to"]
    assert len(points) == 2, f"Expected 2 points_to edges, got {points}"
    by_target = {e["target"]: e for e in points}
    assert by_target[console]["source"] == file_id
    assert by_target[login]["source"] == file_id
    for e in points:
        assert e["confidence"] == "EXTRACTED"
        assert e["source_file"] == str(env)

    # No node for the non-URL secret / localhost value.
    assert "endpoint://localhost" not in nodes
    assert not any(n.startswith("endpoint://") and "UmmjVrTeoCnRWD" in n for n in nodes)


def test_extract_env_endpoints_strips_userinfo_from_dsn_urls(tmp_path):
    """A DSN-style URL (https://<key>@sentry.io/…) must key on the host, not the
    embedded secret — no endpoint id leaks the credential."""
    env = tmp_path / ".env.preview"
    env.write_text(
        "VITE_SENTRY_DSN=https://abc123def@o110884.ingest.sentry.io/6223554\n",
        encoding="utf-8",
    )
    result = extract_env_endpoints(env)
    nodes = {n["id"] for n in result["nodes"]}
    assert "endpoint://o110884.ingest.sentry.io" in nodes
    assert not any("abc123def" in nid for nid in nodes)


def test_extract_env_endpoints_skips_non_env_files():
    from pathlib import Path
    assert not is_env_endpoint_file(Path("config.py"))
    assert not is_env_endpoint_file(Path("package.json"))
