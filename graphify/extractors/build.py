from __future__ import annotations

import shlex
from pathlib import Path

from graphify.extractors.images import image_ref_node


def extract_build(path: Path) -> dict:
    if path.name.lower() == "dockerfile":
        return _extract_dockerfile(path)
    raise NotImplementedError


def _extract_dockerfile(path: Path) -> dict:
    """Parse a Dockerfile's FROM lines into build-stage and image nodes."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_image_ids: set[str] = set()
    stage_counter = 0

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("FROM "):
            continue
        tokens = shlex.split(stripped[len("FROM ") :])
        ref = None
        alias = None
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "AS" and i + 1 < len(tokens):
                alias = tokens[i + 1]
                break
            if token.startswith("-"):
                i += 1
                continue
            ref = token
            i += 1
        if ref is None or "${" in ref:
            continue
        img_node = image_ref_node(ref)
        if img_node is None:
            continue

        img_id = img_node["id"]
        if img_id not in seen_image_ids:
            seen_image_ids.add(img_id)
            nodes.append(dict(img_node))

        if alias is None:
            alias = img_node["label"].rsplit("/", 1)[-1]
        stage_id = f"build://{path.parent.as_posix()}/Dockerfile/stage{stage_counter}"
        stage_counter += 1
        nodes.append(
            {
                "id": stage_id,
                "label": alias,
                "file_type": "build",
                "source_file": str(path),
                "attributes": {},
            }
        )
        edges.append(
            {
                "source": stage_id,
                "target": img_id,
                "relation": "builds",
                "confidence": "EXTRACTED",
                "source_file": str(path),
            }
        )

    return {"nodes": nodes, "edges": edges, "k8s_candidates": []}
