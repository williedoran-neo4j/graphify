# Graph-RAG over the uber-graph (neo4j-graphrag)

How to run retrieval + generation over the stitched 6-repo graph in Neo4j using
[`neo4j-graphrag-python`](https://github.com/neo4j/neo4j-graphrag-python).

## What's in Neo4j

The uber-graph (built by `graphify extract` → `global add` → `export neo4j --push`)
lands in the `neo4j` database as:

| in Neo4j | value |
|---|---|
| nodes | ~214k |
| relationships | ~595k (native types: `RUNS`, `BUILDS`, `DEPENDS_ON`, `DEFINES`, `SELECTS`, `ROUTES`, `ISSUES`, `SERVES`, `POINTS_TO`, `CALLS`, `IMPORTS`, `REFERENCES`, …) |
| `:Embedded` nodes | nodes that carry a vector |
| vector indexes | `graphify_text_embeddings`, `graphify_code_embeddings` |

Both vector indexes are **1536-dim**, cosine, on the `:Embedded` label:

- `graphify_text_embeddings` on property `embedding_text` — nodes whose
  `file_type` maps to the *text* space (`document`, `paper`, `rationale`,
  `concept`).
- `graphify_code_embeddings` on property `embedding_code` — `code` nodes.

Every node has an `id` property of the form `repo::local_id` (e.g.
`neo4j-cloud::k8s://default/Deployment/foo`), plus a human `label`.

## 1. Embed the query with the SAME embedder

The embeddings were produced with OpenAI `text-embedding-3-small` (1536-dim). To
make a query vector comparable, embed the query with the same model:

```python
from openai import OpenAI
client = OpenAI()

def embed(q: str) -> list[float]:
    return client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
```

## 2. Vector retrieval across the whole corpus

`VectorRetriever` targets a standard `node.embedding` index; our index is
`embedding_text`/`embedding_code`, so use `VectorCypherRetriever` with an
explicit retrieval query:

```python
import neo4j
from neo4j_graphrag.retrievers import VectorCypherRetriever

URI = "bolt://localhost:7687"
driver = neo4j.GraphDatabase.driver(URI, auth=("neo4j", "password"))

RETRIEVAL_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
YIELD node, score
RETURN node.id AS id, node.label AS label, node.repo AS repo, score
"""

retriever = VectorCypherRetriever(
    driver,
    index_name="graphify_code_embeddings",   # or graphify_text_embeddings
    retrieval_query=RETRIEVAL_QUERY,
    neo4j_database="neo4j",
)

res = retriever.search(query_vector=embed("how does text become a cypher query?"), top_k=5)
for item in res.items:
    print(f"{item.content}")   # id / label / repo / score rows
```

> `db.index.vector.queryNodes` is deprecated on newer Neo4j (it emits a warning on
> some builds); the equivalent is `db.index.vector.queryNodes` → use the
> `SEARCH`/`VECTOR` vec-search when the server drops it. See
> [[neo4j-querynodes-deprecated]] in the project memory.

## 3. Graph-RAG: walk from a hit, not just return it

The whole point of the uber-graph is the *edges*. After vector-retrieving a seed
node, expand its neighbourhood along the infra/package relations and hand the
subgraph to an LLM for a grounded answer:

```python
from neo4j_graphrag.generation import GraphRAG

graph_rag = GraphRAG(retriever=retriever)  # retriever = the VectorCypherRetriever above
answer = graph_rag.search(
    query_text="Which service runs the image ghcr.io/org/app, and what builds it?",
    top_k=5,
)
print(answer.answer)
```

Or hand-roll a one-hop expansion and pass the Cypher result as context:

```cypher
// vertical slice: what builds / publishes / runs a given image
MATCH (img {id: $image_id})<-[r:RUNS|BUILDS|PUBLISHES]-(producer)
RETURN type(r) AS relation, producer.label AS producer, producer.repo AS repo
```

## 4. "two schema generators" query (the worked example)

The two frontend entry points differ: **local-file schema generation** stays in
`upx`, while **data-source extraction** reaches `kg-builder`. With the
`POINTS_TO` bridge in place:

```cypher
// frontend -> the backend host it declares
MATCH (env)-[:POINTS_TO]->(ep)
WHERE ep.id CONTAINS 'endpoint://'
  AND env.repo = 'upx'
RETURN env.label AS env_file, ep.id AS backend_host
```

```cypher
// the backend package the host implies (kg-builder depends_on neo4j-graphrag)
MATCH (a:Embedded)-[:DEPENDS_ON]->(b:Embedded)
WHERE a.repo = 'kg-builder'
RETURN a.label, b.label, b.repo
LIMIT 20
```

Combine the two (endpoint host `genai-api.neo4j.io` → the component that serves
it) once a host→ingress join exists; today the `endpoint://` node is the
deterministic join key.

## 5. Retrieval result items

`VectorCypherRetriever` returns `RetrieverResultItem`s with `.content` (the row)
and `.metadata["score"]`. The `return_properties` / `RETRIEVAL_QUERY` shape above
surfaces `id`/`label`/`repo`/`score` so the generator and the human both get the
source-repo provenance.
