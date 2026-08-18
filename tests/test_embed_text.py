"""C1.1 — build_node_text is a pure, deterministic text-family selector.

Contract C1 (R1 thread form): for a text-family node (document/paper/rationale/
concept) it returns exactly ``f"{label}\n{source_file}"``; for ``image`` it
returns ``None`` (excluded from every space). Invariant I2: two runs over an
unchanged node produce byte-identical strings.

The node input is the per-node attrs dict that NetworkX carries on
``G.nodes[nid]`` — id, label, source_file, file_type, and optionally rationale.
"""
from __future__ import annotations

import pytest

from graphify.embed import build_node_text


@pytest.fixture
def document_node() -> dict:
    return {
        "id": "n-doc-01",
        "label": "API Tokens",
        "source_file": "docs/security.md",
        "file_type": "document",
    }


def test_build_node_text_text_family(document_node):
    """T1 — a document node with a non-empty source_file yields exactly
    ``"{label}\n{source_file}"`` — no decoration, no trailing content."""
    assert build_node_text(document_node) == "API Tokens\ndocs/security.md"


def test_build_node_text_deterministic(document_node):
    """T2 — two calls over the same unchanged node are byte-identical (I2)."""
    first = build_node_text(document_node)
    second = build_node_text(document_node)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_image_node_excluded():
    """T3 (first half) — an image node produces no text: build_node_text is None."""
    image_node = {
        "id": "n-IMG-07",
        "label": "architecture diagram",
        "source_file": "assets/arch.png",
        "file_type": "image",
    }
    assert build_node_text(image_node) is None
