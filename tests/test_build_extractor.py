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


def test_extract_build_makefile_targets_and_builds_edges(tmp_path):
    """extract_build on a Makefile or .mk emits one build node per named target,
    and a builds edge (EXTRACTED) when the target's recipe names a concrete
    image. Bare targets get a node but no edge. Image nodes are deduped by id."""
    from graphify.extractors.build import extract_build

    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "build-image:\n"
        "\tdocker build . -t europe-west1-docker.pkg.dev/x/y:tag\n"
        "\n"
        "lint:\n"
        "\tflake8 .\n"
    )

    mkfile = tmp_path / "foo.mk"
    mkfile.write_text(
        "build:\n"
        "\tdocker build . -t quay.io/k/img:latest\n"
    )

    # --- Makefile ---
    result = extract_build(makefile)
    nodes = {n["id"]: n for n in result["nodes"]}
    edges = result["edges"]

    dir_path = tmp_path.as_posix()
    build_img_id = f"makefile://{dir_path}/build-image"
    lint_id = f"makefile://{dir_path}/lint"
    image_id = "image://europe-west1-docker.pkg.dev/x/y"

    # Two build nodes
    assert build_img_id in nodes
    assert lint_id in nodes
    assert nodes[build_img_id]["label"] == "build-image"
    assert nodes[build_img_id]["file_type"] == "build"
    assert nodes[build_img_id]["source_file"] == str(makefile)
    assert nodes[lint_id]["label"] == "lint"
    assert nodes[lint_id]["file_type"] == "build"
    assert nodes[lint_id]["source_file"] == str(makefile)
    assert nodes[build_img_id]["attributes"] == {}
    assert nodes[lint_id]["attributes"] == {}

    # One image node (deduped, source_file=None)
    assert image_id in nodes
    assert nodes[image_id]["file_type"] == "image"
    assert nodes[image_id]["source_file"] is None

    # One builds edge from the target that names an image
    builds_edges = [e for e in edges if e["relation"] == "builds"]
    assert len(builds_edges) == 1
    assert builds_edges[0]["source"] == build_img_id
    assert builds_edges[0]["target"] == image_id
    assert builds_edges[0]["confidence"] == "EXTRACTED"
    assert builds_edges[0]["source_file"] == str(makefile)

    # --- .mk file ---
    result_mk = extract_build(mkfile)
    nodes_mk = {n["id"]: n for n in result_mk["nodes"]}
    edges_mk = result_mk["edges"]

    build_id = f"makefile://{dir_path}/build"
    image_id_mk = "image://quay.io/k/img"

    # One build node
    assert build_id in nodes_mk
    assert nodes_mk[build_id]["label"] == "build"
    assert nodes_mk[build_id]["file_type"] == "build"
    assert nodes_mk[build_id]["source_file"] == str(mkfile)

    # One image node
    assert image_id_mk in nodes_mk
    assert nodes_mk[image_id_mk]["file_type"] == "image"
    assert nodes_mk[image_id_mk]["source_file"] is None

    # One builds edge
    builds_edges_mk = [e for e in edges_mk if e["relation"] == "builds"]
    assert len(builds_edges_mk) == 1
    assert builds_edges_mk[0]["source"] == build_id
    assert builds_edges_mk[0]["target"] == image_id_mk
    assert builds_edges_mk[0]["confidence"] == "EXTRACTED"
    assert builds_edges_mk[0]["source_file"] == str(mkfile)


def test_extract_build_makefile_inline_prerequisites(tmp_path):
    """A Makefile rule in the canonical 'target: prereq ...' form (prerequisites
    on the same line, no trailing colon) must still yield a build node for the
    target and a builds edge when its recipe names a concrete image.
    Prerequisites themselves must NOT become nodes.  Colon-containing lines
    that are not targets (:= assignments, ifeq conditionals) must also produce NO
    node."""
    from graphify.extractors.build import extract_build

    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "build: deps\n"
        "\tdocker build . -t europe-west1-docker.pkg.dev/x/y:tag\n"
        "\n"
        "VAR := x\n"
        "ifeq ($(X),y)\n"
        "\techo ok\n"
        "endif\n"
    )

    result = extract_build(makefile)
    nodes = {n["id"]: n for n in result["nodes"]}
    edges = result["edges"]

    dir_path = tmp_path.as_posix()
    build_id = f"makefile://{dir_path}/build"
    deps_id = f"makefile://{dir_path}/deps"
    image_id = "image://europe-west1-docker.pkg.dev/x/y"

    # Exactly one build node for the target 'build', NOT for 'deps'
    assert build_id in nodes, f"expected node {build_id} in {list(nodes.keys())}"
    assert nodes[build_id]["label"] == "build"
    assert nodes[build_id]["file_type"] == "build"
    assert nodes[build_id]["source_file"] == str(makefile)
    assert deps_id not in nodes, f"prerequisite 'deps' must not become a node"

    # One image node
    assert image_id in nodes
    assert nodes[image_id]["file_type"] == "image"
    assert nodes[image_id]["source_file"] is None

    # One builds edge from target to image
    builds_edges = [e for e in edges if e["relation"] == "builds"]
    assert len(builds_edges) == 1
    assert builds_edges[0]["source"] == build_id
    assert builds_edges[0]["target"] == image_id
    assert builds_edges[0]["confidence"] == "EXTRACTED"
    assert builds_edges[0]["source_file"] == str(makefile)

    # No nodes for := assignment or ifeq conditional lines
    var_node = f"makefile://{dir_path}/VAR"
    ifeq_node = f"makefile://{dir_path}/ifeq"
    assert var_node not in nodes
    assert ifeq_node not in nodes
