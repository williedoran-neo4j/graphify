# Visualizing the uber-graph in Neo4j Bloom

The uber-graph lives in the running Neo4j at `neo4j://localhost:7687` (db
`neo4j`, user `neo4j`). Bloom gives perspective-based graph exploration on top of
it.

## 1. Prerequisite: Neo4j Deskotop + Bloom

- Open **Neo4j Desktop** → connect to the local DBMS (`bolt://localhost:7687`).
- In the DBMS, ensure the **Bloom** plugin is added (Neo4j Desktop → the project
  → the database → "Plugins" → Bloom). The browser at
  `http://localhost:7474/browser/` is the *live* browser, not Bloom; Bloom is a
  separate Desktop pane.

If you only have the Neo4j *browser* (no Desktop), the same exploration works
with browser Cypher + `:style` (see §4).

## 2. Create a perspective (this is where it becomes useful)

Bloom is driven by a **perspective** (a JSON/YAML file: the category-graph of
node categories + the relationships between them). Create one targeted at the
uber-graph's infra spine:

```json
{
  "name": "Uber-graph infra",
  "nodeCategories": [
    {"name": "K8sWorkload", "label": "K8s"},
    {"name": "Image", "label": "Image"},
    {"name": "Service", "label": "K8s", "selector": "n.kind = 'Service'"},
    {"name": "Certificate", "label": "K8s", "selector": "n.kind = 'Certificate'"},
    {"name": "CIJob", "label": "Ci"},
    {"name": "Build", "label": "Build"},
    {"name": "Package", "label": "Code", "selector": "n.type = 'package'"},
    {"name": "Endpoint", "label": "Concept", "selector": "n.id CONTAINS 'endpoint://'"}
  ],
  "relationshipCategories": [
    {"name": "RUNS", "type": "RUNS"},
    {"name": "BUILDS", "type": "BUILDS"},
    {"name": "PUBLISHES", "type": "PUBLISHES"},
    {"name": "DEPENDS_ON", "type": "DEPENDS_ON"},
    {"name": "DEFINES", "type": "DEFINES"},
    {"name": "ROUTES", "type": "ROUTES"},
    {"name": "SELECTS", "type": "SELECTS"},
    {"name": "ISSUES", "type": "ISSUES"},
    {"name": "SERVES", "type": "SERVES"},
    {"name": "POINTS_TO", "type": "POINTS_TO"}
  ]
}
```

Save it, then in Bloom: **Create Perspective** → load the file. Filter to the
relationships you want (e.g. only `RUNS/BUILDS/PUBLISHES`) and search by
`Image`, `Deployment`, or `endpoint://`.

## 3. The infra spine you'll see

```
Image  <-RUNS-  Deployment/StatefulSet (K8s)
  ^
  |-PUBLISHES- CiJob (Ci)
  |-BUILDS-   Build (Makefile/Dockerfile target)
  ^
Service -SELECTS-> Deployment
  ^
Certificate -SERVES-> Service   (ClusterIssuer -ISSUES-> Certificate)
Ingress -ROUTES-> Service
.env (Concept/endpoint) -POINTS_TO-> endpoint://genai-api.neo4j.io
Package (Code) -DEPENDS_ON-> Package (other repo)
```

Bloom will lay these out by category; with the perspective limited to the spine
relations it stays readable (the full 595k-edge corpus is not).

## 4. Without Bloom: browser visualization

In `http://localhost:7474/browser/`, same graph, ad-hoc:

```cypher
// the cross-repo image join
MATCH p=(img:Image)<-[:RUNS|PUBLISHES|BUILDS]-(x)
RETURN p LIMIT 100
```

```cypher
// a vertical TLS + deploy slice
MATCH p=(iss:ClusterIssuer)-[:ISSUES]->(c:Certificate)-[:SERVES]->(s:Service)-[:SELECTS]->(d:Deployment)
RETURN p LIMIT 25
```

Style the node labels with the browser's `:style` (bottom-left) to color by
`repo` or label, e.g.

```
:style
node { defaultCaption: "label"; }
node.Image { color: #4285f4; }
node.K8s { color: #ea4335; }
node.Ci { color: #fbbc04; }
```

## 5. What makes it fast / legible

- The `:GraphifyNode(id)` index (created by the push) keeps `MATCH {id:...}` and
  Bloom search fast.
- The two vector indexes enable Bloom's *phrase/neighbor* search on `:Embedded`.
- Cap node fetches (`LIMIT`) — the corpus is 214k nodes; render in slices, not
  `RETURN *`.
