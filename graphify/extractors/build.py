from __future__ import annotations

import shlex
from pathlib import Path

from graphify.extractors.images import image_ref_node


def extract_build(path: Path) -> dict:
    if path.name.lower() == "dockerfile":
        return _extract_dockerfile(path)
    return _extract_makefile(path)


def _extract_makefile(path: Path) -> dict:
    """Parse a Makefile/.mk file's named targets: one build node per target,
    plus a builds edge per target whose recipe names a concrete image."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_image_ids: set[str] = set()

    cur_target_id: str | None = None
    cur_label: str | None = None
    recipe_tokens: list[str] = []

    def flush() -> None:
        nonlocal cur_target_id, cur_label, recipe_tokens
        if cur_target_id is not None and cur_label is not None:
            img_node = None
            for i, tok in enumerate(recipe_tokens):
                if tok in ("-t", "--tag"):
                    if i + 1 < len(recipe_tokens):
                        img_node = image_ref_node(recipe_tokens[i + 1])
                    break
            if img_node is None:
                for tok in recipe_tokens:
                    img_node = image_ref_node(tok)
                    if img_node is not None:
                        break
            if img_node is not None:
                img_id = img_node["id"]
                if img_id not in seen_image_ids:
                    seen_image_ids.add(img_id)
                    nodes.append(dict(img_node))
                edges.append(
                    {
                        "source": cur_target_id,
                        "target": img_id,
                        "relation": "builds",
                        "confidence": "EXTRACTED",
                        "source_file": str(path),
                    }
                )
        cur_target_id = None
        cur_label = None
        recipe_tokens = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            flush()  # a blank line ends the preceding recipe
            continue
        if line.startswith((" ", "\t")):
            if cur_target_id is not None:
                recipe_tokens.extend(shlex.split(stripped))
            continue
        if ":" in stripped and not stripped.partition(":")[2].startswith(("=", "?", ":")):
            name = stripped.split(":", 1)[0].strip()
            flush()
            # skip Makefile directives (.PHONY etc.) and file inclusions
            if not name or name == "include" or name.startswith("."):
                continue
            cur_target_id = f"makefile://{path.parent.as_posix()}/{name}"
            cur_label = name
            nodes.append(
                {
                    "id": cur_target_id,
                    "label": name,
                    "file_type": "build",
                    "source_file": str(path),
                    "attributes": {},
                }
            )
            continue
        flush()  # a non-recipe top-level statement ends the preceding recipe

    flush()
    return {"nodes": nodes, "edges": edges, "k8s_candidates": []}


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
