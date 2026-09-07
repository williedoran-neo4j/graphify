# Uber-graph setup: clone → extract → stitch → Neo4j

Step-by-step runbook to reproduce the cross-repo "uber graph" end to end, from a
clean machine. Every command below was verified in a live run. The fork carries
three layers of work not yet in upstream `v8`:

1. **YAML/CI/k8s/argo/kustomize extractors** — richer nodes + edges.
2. **Cross-repo join** — `global add` image dedup + a package/code dependency link pass.
3. **Fast Neo4j push** — an `id` range index is created before the edge loop
   (without it, a 100k-node push takes hours instead of minutes).

> **Target branch:** `feat/k8s-yaml-ast` (the fork). All of the above lives here.

---

## 0. Prerequisites

- **Python 3.12** (the venv tests use 3.12; a bare machine `python` may point at a
  3.x without `pytest` — always use `.venv/bin/python`).
- **git**
- **Docker + Docker Compose** (for Neo4j)
- **OpenAI API key** (for the OpenAI embedding variant) — export as `OPENAI_API_KEY`
- **ollama** + the embed models (only for the local variant, section 3 Variant B):
  ```bash
  ollama pull nomic-embed-text      # tiny, 137M, safe
  ollama pull nomic-embed-code      # 7B, heavy — see the warning in Variant B
  ```

---

## 1. Clone the fork + checkout the branch

```bash
git clone git@github.com:williedoran-neo4j/graphify.git
cd graphify
git checkout feat/k8s-yaml-ast
```

(The upstream is `git@github.com:Graphify-Labs/graphify.git`, mainline branch `v8`.)

---

## 2. Set up the local env

Run from the repo rather than `pip install`-ing a released wheel (the released
`graphifyy` does not yet have the work on this branch):

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

Sanity-check:

```bash
.venv/bin/python -m graphify --help
```

> Gotcha: bare `python -m pytest` may resolve to a Python that lacks pytest. Use
> `.venv/bin/python -m pytest` for anything involving tests.

---

## 3. Configure the embedding backends

Embedding is controlled entirely by four env vars. There are three variants.

### Variant A — all OpenAI (recommended, what shipped)

Uniform **1536-dim** across both spaces. Cost ≈ **$1.16** for the six repos +
one global re-embed (~58M tokens @ `text-embedding-3-small` $0.02/Mtok). The
laptop stays idle.

```bash
export GRAPHIFY_EMBED_TEXT_BACKEND=openai
export GRAPHIFY_EMBED_TEXT_MODEL=text-embedding-3-small
export GRAPHIFY_EMBED_CODE_BACKEND=openai
export GRAPHIFY_EMBED_CODE_MODEL=text-embedding-3-small
```

### Variant B — all local ollama

`nomic-embed-text` is 137M (768-dim), `nomic-embed-code` is 7B (3584-dim).

```bash
export GRAPHIFY_EMBED_TEXT_BACKEND=ollama
export GRAPHIFY_EMBED_TEXT_MODEL=nomic-embed-text
export GRAPHIFY_EMBED_CODE_BACKEND=ollama
export GRAPHIFY_EMBED_CODE_MODEL=nomic-embed-code
```

> ⚠️ **Ollama code-embed warning (hit twice in practice):** the 7B
> `nomic-embed-code` model spikes an M-series GPU to ~94% and can **deadlock** the
> local ollama server under load. If you use it: run one repo at a time
> (never two `extract --embed` in parallel), and if it wedges (0% CPU, curl
> `/api/embed` hangs), do `brew services restart ollama` and re-run. Prefer
> Variant A for code embeddings.

### Variant C — mixed (text=OpenAI, code=ollama)

Text space on OpenAI, code space on ollama. Works (spaces have separate indexes),
but the code space is then 3584-dim vs the text space's 1536-dim. Prefer A for
uniformity.

---

## 4. Extract each repo (per-repo graph + embeddings)

For each repo, run (with the Variant env vars exported):

```bash
.venv/bin/python -m graphify extract /path/to/repo --embed
```

This writes, per repo:

- `graphify-out/graph.json` (nodes + edges)
- `graphify-out/embeddings.npz` (text + code vectors)

The six known repos are, e.g.:

```bash
for r in genai-cloud kg-builder neo4j-graphrag-python text2cypher upx neo4j-cloud; do
  .venv/bin/python -m graphify extract "/Users/williedoran/Dev/$r" --embed
done
```

Notes:

- **`--embed`** triggers the dual-space embedding; omit it for a structural graph
  only (no vectors).
- **`--code-only`** skips the semantic LLM doc/paper extraction — no LLM cost, no
  doc nodes; the node/edge structure is still extracted deterministically. Use
  full extraction (no `--code-only`) to also get `document`/`paper` nodes.
- Full extraction charges the LLM `--backend` (e.g. `claude`) for doc/paper
  extraction — observed ~$0.28–$0.57 per large repo.

---

## 5. Stitch the repos into one uber-graph

`global add` merges each repo's graph into `~/.graphify/global-graph.json`,
deduping `source_file=None` image nodes by label and linking package dependencies
cross-repo.

```bash
for r in genai-cloud kg-builder neo4j-graphrag-python text2cypher upx neo4j-cloud; do
  .venv/bin/python -m graphify global add "/Users/williedoran/Dev/$r/graphify-out/graph.json" --as "$r"
done
```

Verify:

```bash
.venv/bin/python -m graphify global list
```

Then re-embed the joined graph (the per-repo `.npz` files are keyed by un-prefixed
ids; the joined graph uses `repo_tag::local_id`, so it needs its own pass). **Set
the same `GRAPHIFY_EMBED_*` vars first** (OpenAI in Variant A):

```bash
.venv/bin/python -m graphify global re-embed
# writes ~/.graphify/embeddings-global.npz
```

---

## 6. Spin up Neo4j

Use the compose that ships with `neo4j-graphrag-python` (in that repo, if you have
it cloned; otherwise create the same file):

```bash
cd /path/to/neo4j-graphrag-python
docker compose -f tests/e2e/docker-compose.yml up -d --wait
```

Minimal equivalent `docker-compose.yml` if you don't have that repo:

```yaml
services:
  neo4j:
    image: neo4j:enterprise
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_server_memory_heap_max__size: 6G
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p password 'RETURN 1' || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 30
      start_period: 20s
```

Credentials: user **`neo4j`**, password **`password`**. Default database is
**`neo4j`**. APOC is installed; **GDS is not** (the plugin list is `["apoc"]`
only) — fine for vector search / graph-RAG.

---

## 7. Publish the uber-graph to Neo4j

First, a one-line gotcha: `global re-embed` wrote `embeddings-global.npz`, but
`export neo4j` reads `embeddings.npz` — symlink it:

```bash
ln -sf ~/.graphify/embeddings-global.npz ~/.graphify/embeddings.npz
```

Then push (password via env to keep it off argv):

```bash
NEO4J_PASSWORD=password .venv/bin/python -m graphify export neo4j \
  --graph ~/.graphify/global-graph.json \
  --push bolt://localhost:7687 --user neo4j
```

This MERGEs all nodes + edges (idempotent — safe to re-run) and creates two
vector indexes on the `:Embedded` label:
`graphify_text_embeddings` and `graphify_code_embeddings`.

> The `export neo4j --push` is per-edge MERGE (no UNWIND batching). On **this
> branch** (`feat/k8s-yaml-ast`, commit `2a9ca71`) it creates a
> `:GraphifyNode(id)` range index before the edge loop, which is what makes it
> minutes rather than hours. Still, a ~265k-edge graph takes a few minutes.

---

## 8. Verify

Open Neo4j Browser and log in:

- URL: **http://localhost:7474/browser/**
- user `neo4j` / password `password`, database `neo4j`

Sample Cypher:

```cypher
// total sizes
MATCH (n) RETURN count(n) AS nodes;
MATCH ()-[r]->() RETURN count(r) AS relationships;

// the cross-repo/infra stitch edges
MATCH p=()-[r:RUNS|PUBLISHES|BUILDS|DEPENDS_ON]->() RETURN p LIMIT 100;

// a vector search (after the data is indexed)
CALL db.index.vector.queryNodes('graphify_code_embeddings', 5, '') YIELD node RETURN node LIMIT 5;
```

Expected result: **~100k nodes, ~265k relationships, ~98k embedded nodes**, with
`graphify_text_embeddings` and `graphify_code_embeddings` both `ONLINE`
(`SHOW INDEXES`).

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| ollama embed wedges (0% CPU, `/api/embed` hangs) | The 7B `nomic-embed-code` deadlocked. `brew services restart ollama`, re-run one repo at a time, or switch code embeddings to OpenAI (Variant A). |
| `global re-embed` wrote `embeddings-global.npz` but push found no vectors | The `export neo4j` path looks for `embeddings.npz`. `ln -sf ~/.graphify/embeddings-global.npz ~/.graphify/embeddings.npz` (section 7). |
| Neo4j push is taking hours | You are on an old branch without the `:GraphifyNode(id)` range index (the edge MATCH is a full scan per edge). Use `feat/k8s-yaml-ast` / commit `2a9ca71`. |
| `graphify extract` says model returned a hollow response | Transient LLM rate-limit/refusal; files are marked for re-extraction and retried next run. |
| bare `pytest` fails with "No module named pytest" | Use `.venv/bin/python -m pytest`. |
