# graphify-fork — Repository Index

A codebase map for navigation. The library turns source trees + docs into a
knowledge graph (`graph.json`), enriches it with **AST extraction** and **vector
embeddings**, and serves it over an MCP server for semantic + graph query.

> Auto-generated 2026-08-25. Line counts are approximate. The `.claude/tdd/`
> plan/progress/contracts state is gitignored and omitted here — it is the
> TDD orchestrator's shared memory, not shipped source.

---

## 1. The core pipeline (top-to-bottom)

| Step | Module | What it does |
|---|---|---|
| **Detect** | `detect.py` (2326) | Scan a target dir; classify files by language/type |
| **Extract (AST)** | `extract.py` (6831) + `extractors/` | tree-sitter structural extraction → `nodes` + `edges` dicts |
| **Build** | `build.py` (2044) | Assemble node+edge dicts into a NetworkX graph (dedup + direction) |
| **Cluster** | `cluster.py` (320) | Community detection (Leiden → Louvain fallback) |
| **Analyze** | `analyze.py` (749) | god-nodes, cross-community "surprising connections", suggested questions |
| **Embed (vectors)** | `embed.py` (450) + `embeddings.py` + `search.py` (206) | Embed nodes into `text`/`code` spaces → `embeddings.npz` sidecar |
| **Export** | `export.py` (1194) + `exporters/` (graphdb, html) | Write graph.json + graphdb variants + HTML |
| **Serve** | `serve.py` (2520) | MCP server: `query_graph`, `semantic_search`, `get_node`, `get_neighbors`, … |

**AST → vector embedding (the code path):** `extractors/*.py` (tree-sitter per-
language) emit `code` `file_type` nodes; `embed.py::build_node_text` reads the
original `source_file` content for those nodes; `enrich_embeddings` partitions
nodes by `_embed_space` (`code` → `nomic-embed-code` [3584-dim], text-family →
`nomic-embed-text` [768-dim]) and writes a two-space `embeddings.npz` sidecar
next to `graph.json`. Query side passes through `search.py::_real_query_embed`.

---

## 2. Entry points

| File | Role |
|---|---|
| `graphify/__main__.py` (717) | CLI: reads `sys.argv[1]`, universal `--help` guard, dispatches |
| `graphify/cli.py` (4288) | `dispatch_command`: every non-install subcommand |
| `graphify/install.py` (2296) | `install`/`uninstall` of skills into IDEs/agents |
| `graphify/serve.py` | `python -m graphify.serve` — MCP over stdio or `--transport http` |
| `pyproject.toml` | Project metadata + entry points + extras (incl. `[mcp]`) |

---

## 3. Source modules (by area)

### Extraction & ingestion
- `graphify/extract.py` — deterministic tree-sitter extraction, headers/edges
- `graphify/extractors/` — one file per language (go, rust, python via engine,
  csharp, ts, sql, terraform, … ~30) + `engine.py`, `base.py`, `models.py`
- `graphify/detect.py` — file classification (code vs doc vs paper vs image)
- `graphify/ingest.py`, `manifest_ingest.py`, `mcp_ingest.py`, `scip_ingest.py`,
  `pg_introspect.py`, `cargo_introspect.py`, `google_workspace.py` — alternate
  ingest sources (Git, MCP, SCIP, Postgres, Cargo, Google Workspace)

### Graph assembly & structure
- `graphify/build.py` — node/edge dicts → NetworkX graph (dedup, direction)
- `graphify/dedup.py` (979) — entity dedup pipeline
- `graphify/ids.py` — node identity
- `graphify/cluster.py` — community detection
- `graphify/global_graph.py` — cross-repo global graph (`global add`/`re-embed`)
- `graphify/multigraph_compat.py`, `resolver_registry.py`,
  `symbol_resolution.py`, `pascal_resolution.py`, `ruby_resolution.py`,
  `file_slice.py` — symbol/file resolution helpers

### Vectors / embeddings / search
- `graphify/embed.py` — text+code sidecar writer (`enrich_embeddings`), cache
- `graphify/embeddings.py` — backend registry (`ollama`/`openai`) + live routing
- `graphify/search.py` — `search_vectors` (cosine over unit rows), query-embed seam
- `graphify/cache.py` (1488) — cache dirs incl. per-`(backend,model)` embedding cache

### Querying & MCP
- `graphify/serve.py` — MCP tools (`query_graph`, `semantic_search`, `get_node`,
  `get_neighbors`, `get_community`, `god_nodes`, …), stdio + HTTP transports
- `graphify/affected.py` — reverse traversal to nodes impacted by X
- `graphify/querylog.py`, `reflect.py`, `wiki.py` — query logging / reflections / wiki

### LLM backend
- `graphify/llm.py` (3173) — backend registry, `detect_backend`, community naming

### Output & rendering
- `graphify/export.py` + `exporters/` (graphdb → Neo4j/FalkorDB, html)
- `graphify/callflow_html.py` (2051), `tree_html.py` (603), `report.py`
- `graphify/analyze.py` — analysis text/json

### Infra / cross-cutting
- `graphify/paths.py` — single source of truth for the `graphify-out/` dir name
- `graphify/manifest.py`, `security.py`, `validate.py`, `hooks.py`,
  `diagnostics.py`, `prs.py`, `benchmark.py`, `transcribe.py`,
  `semantic_cleanup.py`, `minhash.py` (`_minhash.py`), `watch.py`

---

## 4. Tests

`tests/` — 224 modules (~4760 tests, 7 skipped at last run). Notable homes by
area:
- `test_embed*.py` — embed/vector sidecar/backends/cache/search (R1–R11)
- `test_query_graph_blend.py` — the semantic blend (R9)
- `test_serve*.py`, `test_query_cli.py`, `test_extract_cli.py`, `test_affected_cli.py` — MCP + CLI
- `test_embed_cli.py`, `test_extract_embed_flag.py` — CLI embed commands (R10)
- `test_graphdb*.py`, `test_neo4j*.py`, `test_falkordb*.py` — Neo4j/FalkorDB tier (R7)

---

## 5. Key docs

- `README.md` (60K) — full user guide incl. MCP config + `--transport http`
- `ARCHITECTURE.md` — design overview
- `BENCHMARKS.md` — perf notes
- `CHANGELOG.md` — history
- `docs/` — `how-it-works.md`, `docker-mcp-sqlite.md`, `node-summaries-rfc.md`,
  `specs/` (dated steel-thread specs), `superpowers/`, `translations/`
- `.claude/` — TDD orchestrator (agents, skills, `tdd/` state) — gitignored

---

## 6. Config / tooling

- `pyproject.toml` — package + deps (via `uv`; `uv.lock` pinned)
- `.pre-commit-config.yaml` — ruff + hooks
- `Dockerfile`, `.dockerignore` — container build
- `.github/` — CI workflows

---

## 7. Feature map (R1–R11, the embedding-enrichment line)

The `.claude/tdd/progress.md` ledger tracks requirements R1–R11 (all done):

```
R1–R2   text-space .npz sidecar + semantic_search tool
R3      real text embedding backend (batching, LRU, routing)
R4      embedding write-cache (incremental guarantee)
R5      rich deterministic node-text construction
R6      global-graph ordering & identity
R7      Neo4j tier (props + :Embedded + vector indexes)
R8      second vector space (code model) + dual-space search
R9      query_graph semantic blend (gated, lexical priority)
R10     CLI: 'embed' + 'extract --embed'
R11     real query-side embedding for the MCP read tools
```
