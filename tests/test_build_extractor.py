def test_build_file_type_passes_validation_and_survives_build():
    """A build node with file_type='build' must pass schema validation and
    retain its file_type through graph assembly, not be coerced to 'concept'."""
    from graphify.validate import validate_extraction
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "makefile://x/y",
                "label": "y",
                "file_type": "build",
                "source_file": "Makefile",
                "attributes": {},
            }
        ],
        "edges": [],
    }

    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'build'" in e]
    assert file_type_errors == []

    G = build_from_json(extraction)
    assert G.nodes["makefile://x/y"]["file_type"] == "build"


def test_extract_build_dockerfile_from_parsing(tmp_path):
    """extract_build on a Dockerfile emits one stage node per FROM line and a
    builds edge to the image node for each concrete image reference.

    Stage node ids are stable, slash-joined, and carry no repo identity:
    build://{parent_dir}/Dockerfile/stage{N}. The label is the AS alias if
    present, otherwise the base image name."""
    from graphify.extractors.build import extract_build

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM quay.io/keycloak/keycloak:19.0.3\n"
        "FROM quay.io/k/registry1:v AS build\n"
        "FROM quay.io/k/registry2:v\n"
        "FROM ${BASE_IMAGE}\n"
        "FROM --platform=linux/amd64 registry.example.com/img:v1 AS s\n"
        "FROM ${REGISTRY}/img:v2\n"
    )

    result = extract_build(dockerfile)

    nodes = {n["id"]: n for n in result["nodes"]}
    edges = result["edges"]

    stage0_id = f"build://{tmp_path.as_posix()}/Dockerfile/stage0"
    stage1_id = f"build://{tmp_path.as_posix()}/Dockerfile/stage1"
    stage2_id = f"build://{tmp_path.as_posix()}/Dockerfile/stage2"
    stage3_id = f"build://{tmp_path.as_posix()}/Dockerfile/stage3"
    img0_id = "image://quay.io/keycloak/keycloak"
    img1_id = "image://quay.io/k/registry1"
    img2_id = "image://quay.io/k/registry2"
    img3_id = "image://registry.example.com/img"

    # Four stage nodes (concrete FROMs only; ${BASE_IMAGE} and ${REGISTRY}/img skipped)
    assert stage0_id in nodes
    assert stage1_id in nodes
    assert stage2_id in nodes
    assert stage3_id in nodes
    assert nodes[stage0_id]["file_type"] == "build"
    assert nodes[stage1_id]["file_type"] == "build"
    assert nodes[stage2_id]["file_type"] == "build"
    assert nodes[stage3_id]["file_type"] == "build"
    assert nodes[stage0_id]["source_file"] == str(dockerfile)
    assert nodes[stage1_id]["label"] == "build"  # AS alias
    assert nodes[stage2_id]["label"] == "registry2"  # image name
    assert nodes[stage3_id]["label"] == "s"  # AS alias after --platform flag

    # Four image nodes produced by image_ref_node
    assert img0_id in nodes
    assert img1_id in nodes
    assert img2_id in nodes
    assert img3_id in nodes
    assert nodes[img0_id]["file_type"] == "image"
    assert nodes[img0_id]["source_file"] is None

    # Four builds edges: stage -> image
    builds_edges = [e for e in edges if e["relation"] == "builds"]
    assert len(builds_edges) == 4
    assert all(e["confidence"] == "EXTRACTED" for e in builds_edges)
    assert all(e["source_file"] == str(dockerfile) for e in builds_edges)

    targets = {(e["source"], e["target"]) for e in builds_edges}
    assert (stage0_id, img0_id) in targets
    assert (stage1_id, img1_id) in targets
    assert (stage2_id, img2_id) in targets
    assert (stage3_id, img3_id) in targets
