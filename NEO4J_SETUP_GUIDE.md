# 📊 Neo4j Graph Database Integration Guide

## Overview

This guide shows how to integrate **Neo4j** as your graph database for storing documents, relationships, and supporting advanced graph queries in your GraphRAG fine-tuning pipeline.

**Neo4j Benefits:**
- Native graph storage and querying
- Powerful Cypher query language
- ACID compliance
- Real-time relationship queries
- Built-in path finding and traversal

---

## 🚀 Quick Start (5 Minutes)

### 1. Get Neo4j Cloud Credentials

Visit [Neo4j Aura](https://neo4j.com/cloud/aura-free/) and create a free cloud instance:

```
✓ Go to https://neo4j.com/cloud/aura-free/
✓ Sign up for free
✓ Create new database (takes ~2 minutes)
✓ Copy connection details:
  - Neo4j URI (e.g., neo4j+s://your-db.databases.neo4j.io:7687)
  - Username (default: neo4j)
  - Password (set during creation)
  - Database name (default: neo4j)
```

### 2. Update .env File

```bash
# .env

# Existing credentials
GOOGLE_API_KEY=your_api_key_here

# Add these Neo4j lines
NEO4J_URI=neo4j+s://your-db.databases.neo4j.io:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
NEO4J_DATABASE=neo4j
```

### 3. Install Dependencies

```bash
pip install neo4j==5.14.0
```

Or use the updated requirements:
```bash
pip install -r requirements.txt
```

### 4. Test Connection

```bash
python -c "
from data_loaders import setup_neo4j_from_env
db = setup_neo4j_from_env()
if db:
    stats = db.get_graph_stats()
    print('✓ Neo4j connected!')
    print(f'Stats: {stats}')
    db.close()
"
```

---

## 🏗️ Neo4j Architecture

```
┌─────────────────────────────────────┐
│      FireRisk Documents             │
│  (5,000 images + metadata)          │
└──────────────┬──────────────────────┘
               │
               ↓
    ┌──────────────────────┐
    │  Document Nodes      │
    │  - id                │
    │  - content           │
    │  - label (0-6)       │
    │  - created_at        │
    └──────────┬───────────┘
               │
               ↓ (via create_graph_edges_from_documents)
    ┌──────────────────────┐
    │  Relationships       │
    │  - SIMILAR_TO        │
    │  - SAME_CLASS        │
    │  - GEOGRAPHICALLY... │
    └──────────┬───────────┘
               │
               ↓
    ┌──────────────────────┐
    │    Neo4j Database    │
    │  (Full graph store)  │
    └──────────┬───────────┘
               │
               ↓
    ┌──────────────────────┐
    │   Graph Queries      │
    │  - Find similar docs │
    │  - Traverse paths    │
    │  - Analyze clusters  │
    └──────────────────────┘
```

---

## 💻 Code Examples

### Example 1: Basic Connection

```python
from data_loaders import Neo4jGraphDatabase

# Connect to Neo4j
db = Neo4jGraphDatabase(
    uri="neo4j+s://your-db.databases.neo4j.io:7687",
    username="neo4j",
    password="your-password"
)

# Use the database
stats = db.get_graph_stats()
print(f"Total documents: {stats['total_nodes']}")
print(f"Total relationships: {stats['total_relationships']}")

# Clean up
db.close()
```

### Example 2: Using Environment Variables

```python
from data_loaders import setup_neo4j_from_env

# Auto-load from .env
db = setup_neo4j_from_env()

if db:
    stats = db.get_graph_stats()
    print(stats)
    db.close()
else:
    print("Neo4j not configured")
```

### Example 3: Create Documents and Relationships

```python
from data_loaders import setup_neo4j_from_env, create_documents_with_neo4j

db = setup_neo4j_from_env()

# Prepare documents
documents = [
    {"id": "doc1", "content": "Fire risk assessment", "label": 0},
    {"id": "doc2", "content": "Forest conditions", "label": 1},
    {"id": "doc3", "content": "Weather patterns", "label": 0},
]

# Define relationships
edges = [
    ("doc1", "doc2", "RELATED_TO"),
    ("doc1", "doc3", "SAME_CLASS"),
    ("doc2", "doc3", "GEOGRAPHIC_PROXIMITY"),
]

# Create in Neo4j
result = create_documents_with_neo4j(documents, edges, db)

print(f"Documents created: {result['documents_created']}")
print(f"Relationships created: {result['relationships_created']}")
print(f"Graph stats: {result['graph_stats']}")

db.close()
```

### Example 4: Query Related Documents

```python
from data_loaders import setup_neo4j_from_env

db = setup_neo4j_from_env()

# Find all documents related to a specific document
related = db.query_similar_documents("doc1", limit=5)

for doc in related:
    print(f"ID: {doc['id']}, Content: {doc['content']}")

# Find documents by specific relationship type
similar = db.query_similar_documents(
    "doc1",
    relation_type="SAME_CLASS",
    limit=10
)

db.close()
```

### Example 5: Integration with Multi-Agent Pipeline

```python
from data_loaders import (
    FireRiskLoader,
    create_graph_edges_from_documents,
    setup_neo4j_from_env,
    create_documents_with_neo4j
)
from fine_tune import run_multi_agent_pipeline, get_default_config

# Load FireRisk data
loader = FireRiskLoader()
docs = loader.download(limit=1000)

# Create graph edges
edges = create_graph_edges_from_documents(docs, 0.8)

# Store in Neo4j
db = setup_neo4j_from_env()
if db:
    result = create_documents_with_neo4j(docs, edges, db)
    print(f"Stored {result['documents_created']} documents in Neo4j")
    
    # Get graph stats for monitoring
    stats = db.get_graph_stats()
    print(f"Graph contains: {stats['total_nodes']} nodes, {stats['total_relationships']} edges")
    
    db.close()

# Run training with graph-backed data
config = get_default_config()
results = run_multi_agent_pipeline(
    documents=docs,
    edges=edges,
    config=config
)

print("Training complete!")
```

---

## 📊 Neo4j API Reference

### Neo4jGraphDatabase Class

#### Methods

##### `__init__(uri, username, password, database="neo4j")`
Create connection to Neo4j database.

```python
db = Neo4jGraphDatabase(
    uri="neo4j+s://your-db.databases.neo4j.io:7687",
    username="neo4j",
    password="your-password"
)
```

##### `create_document_nodes(documents: List[Dict]) -> int`
Create document nodes in Neo4j.

**Parameters:**
- `documents`: List of dicts with keys: `id`, `content`, `label`

**Returns:** Number of nodes created

**Example:**
```python
docs = [
    {"id": "1", "content": "text", "label": 0},
    {"id": "2", "content": "text", "label": 1},
]
count = db.create_document_nodes(docs)
```

##### `create_relationships(edges: List[Tuple[str, str, str]]) -> int`
Create relationships between documents.

**Parameters:**
- `edges`: List of (source_id, target_id, relation_type) tuples

**Returns:** Number of relationships created

**Example:**
```python
edges = [
    ("1", "2", "SIMILAR_TO"),
    ("2", "3", "SAME_CLASS"),
]
count = db.create_relationships(edges)
```

##### `query_similar_documents(doc_id: str, relation_type: str = None, limit: int = 5) -> List[Dict]`
Query documents related to a given document.

**Parameters:**
- `doc_id`: Document ID to query
- `relation_type`: Optional specific relationship type
- `limit`: Maximum results to return

**Returns:** List of related document dicts

**Example:**
```python
# Get all related documents
related = db.query_similar_documents("doc1", limit=10)

# Get only documents with SIMILAR_TO relationship
similar = db.query_similar_documents(
    "doc1",
    relation_type="SIMILAR_TO",
    limit=5
)
```

##### `get_graph_stats() -> Dict`
Get statistics about the graph.

**Returns:** Dict with:
- `total_nodes`: Number of document nodes
- `total_relationships`: Number of relationships
- `label_distribution`: Count of documents by label

**Example:**
```python
stats = db.get_graph_stats()
print(stats)
# Output:
# {
#   'total_nodes': 1000,
#   'total_relationships': 5230,
#   'label_distribution': {0: 150, 1: 200, ...}
# }
```

##### `clear_all() -> bool`
⚠️ **WARNING**: Delete all nodes and relationships!

**Returns:** True if successful, False otherwise

**Example:**
```python
if db.clear_all():
    print("Database cleared")
```

##### `close()`
Close database connection.

**Example:**
```python
db.close()
```

---

## 🔒 Security Best Practices

### 1. Never Commit Passwords
Always use environment variables:

```bash
# ✓ GOOD - Use .env
NEO4J_PASSWORD=your-password

# ✗ BAD - Hardcoded
db = Neo4jGraphDatabase(..., password="hardcoded-password")
```

### 2. Use .gitignore
```bash
# .gitignore
.env
.env.local
.env.*.local
```

### 3. Change Default Password
After creating Neo4j Aura instance, change the default password immediately.

### 4. Use Network Security
- Use `neo4j+s://` (secure) not `neo4j://` (unsafe)
- Enable IP whitelisting in Neo4j Aura console
- Use VPN for production

### 5. Minimal Permissions
Create read-only user for fine-tuning if possible:
```cypher
CREATE USER readuser SET PASSWORD 'password' CHANGE REQUIRED false;
GRANT TRAVERSE ON GRAPH neo4j TO readuser;
GRANT MATCH {*} ON GRAPH neo4j TO readuser;
```

---

## 🐛 Troubleshooting

### Connection Failed
```
Error: Failed to connect to Neo4j: ...
```

**Solutions:**
1. Verify URI is correct (includes `neo4j+s://` protocol)
2. Check username and password
3. Confirm database is running (check Neo4j Aura console)
4. Verify firewall allows outbound connections
5. Test with: `python VERIFY_SETUP.py`

### Driver Already Closed
```
Error: Illegal state. Driver is already closed.
```

**Solution:**
```python
# Make sure to call close() only once
db.close()  # ✓ Do this once

db.close()  # ✗ Don't do this twice
db.close()
```

### Out of Memory
```
Error: OutOfMemoryError
```

**Solutions:**
1. Limit documents when creating nodes
2. Clear old data: `db.clear_all()`
3. Use Neo4j Aura paid tier for more resources

### Timeout on Query
```
Error: ClientError: Transaction timed out
```

**Solutions:**
1. Reduce batch size
2. Use `LIMIT` in queries
3. Create indexes on frequently queried fields
4. Check Neo4j server performance

### Package Not Found
```
Error: ModuleNotFoundError: No module named 'neo4j'
```

**Solution:**
```bash
pip install neo4j==5.14.0
```

---

## 🚀 Advanced Usage

### Custom Cypher Queries

Access the underlying Neo4j driver for custom queries:

```python
from data_loaders import setup_neo4j_from_env

db = setup_neo4j_from_env()

# Run custom Cypher query
with db.driver.session(database=db.database) as session:
    result = session.run("""
        MATCH (d:Document)-[r]->(related:Document)
        WHERE d.label = $label
        RETURN d.id as id, count(r) as relationship_count
        ORDER BY relationship_count DESC
        LIMIT 10
    """, label=0)
    
    for record in result:
        print(f"Doc {record['id']}: {record['relationship_count']} relationships")

db.close()
```

### Create Indexes for Performance

```python
with db.driver.session(database=db.database) as session:
    # Create index on document ID for faster lookups
    session.run("CREATE INDEX doc_id IF NOT EXISTS FOR (d:Document) ON (d.id)")
    
    # Create index on label for class-based queries
    session.run("CREATE INDEX doc_label IF NOT EXISTS FOR (d:Document) ON (d.label)")

db.close()
```

### Graph Analytics

```python
with db.driver.session(database=db.database) as session:
    # Find most connected documents
    result = session.run("""
        MATCH (d:Document)-[r]-(other:Document)
        RETURN d.id as id, d.label as label, count(r) as degree
        ORDER BY degree DESC
        LIMIT 20
    """)
    
    for record in result:
        print(f"Document {record['id']} (Label {record['label']}): {record['degree']} connections")

db.close()
```

---

## 📈 Performance Tips

### 1. Batch Operations
```python
# ✗ Slow - one at a time
for doc in documents:
    db.create_document_nodes([doc])

# ✓ Fast - all at once
db.create_document_nodes(documents)
```

### 2. Use Connection Pooling
Neo4j driver handles this automatically. Just reuse the same driver instance.

### 3. Create Indexes
```python
with db.driver.session() as session:
    session.run("CREATE INDEX IF NOT EXISTS ON :Document(id)")
```

### 4. Use Appropriate Limits
```python
# ✓ Good - limits result size
related = db.query_similar_documents(doc_id, limit=10)

# ✗ Bad - no limit, could return thousands
related = db.query_similar_documents(doc_id)
```

---

## 🔄 Integration with Existing Code

### In finetune_setup.py

The setup script can now create Neo4j schema:

```python
# After loading data
neo4j_db = setup_neo4j_from_env()
if neo4j_db:
    result = create_documents_with_neo4j(docs, edges, neo4j_db)
    print(f"Neo4j: {result['documents_created']} documents stored")
```

### In multi_agent_orchestration.py

Agents can use Neo4j for retrieval:

```python
from data_loaders import setup_neo4j_from_env

class RetrieverConfigAgent(Agent):
    def execute(self):
        # Initialize Neo4j if configured
        self.neo4j_db = setup_neo4j_from_env()
        
        if self.neo4j_db:
            stats = self.neo4j_db.get_graph_stats()
            self.results['neo4j_stats'] = stats
```

---

## 📚 Further Learning

### Neo4j Resources
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/)
- [Neo4j Aura](https://neo4j.com/cloud/aura-free/)
- [Graph Database Concepts](https://neo4j.com/developer/graph-database/)

### Related Tools
- [Neo4j Browser](https://neo4j.com/developer/neo4j-browser/) - GUI for exploring graphs
- [Neo4j Desktop](https://neo4j.com/download/) - Local development
- [Neo4j GraphQL](https://neo4j.com/docs/graphql-manual/) - GraphQL support

---

## ✅ Verification Checklist

- [ ] Neo4j account created (free tier)
- [ ] Database deployed and running
- [ ] Connection credentials noted (URI, username, password)
- [ ] .env file updated with Neo4j credentials
- [ ] `neo4j` package installed (`pip install neo4j`)
- [ ] Connection test passed (`python VERIFY_SETUP.py`)
- [ ] Can create document nodes
- [ ] Can query documents
- [ ] Ready to integrate with fine-tuning

---

## 🎯 Next Steps

1. **Setup Neo4j**: Create free Aura instance
2. **Configure**: Update .env with credentials
3. **Test**: Run `python VERIFY_SETUP.py`
4. **Integrate**: Use `setup_neo4j_from_env()` in your code
5. **Monitor**: Check `db.get_graph_stats()` regularly

---

**Happy graph querying! 📊**
