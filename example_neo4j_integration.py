"""
Example: Using Neo4j Graph Database with Multi-Agent Fine-Tuning Pipeline

This example shows how to:
1. Load FireRisk data
2. Create graph edges
3. Store everything in Neo4j
4. Query the graph during training
5. Run the multi-agent pipeline
"""

import json
from typing import Dict, List
from data_loaders import (
    FireRiskLoader,
    create_graph_edges_from_documents,
    setup_neo4j_from_env,
    create_documents_with_neo4j,
    create_firetask_setup
)
from fine_tune import run_multi_agent_pipeline, get_default_config


def example_neo4j_basic():
    """
    Example 1: Basic Neo4j connection and querying
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Neo4j Connection")
    print("=" * 60)
    
    # Connect to Neo4j from .env
    db = setup_neo4j_from_env()
    
    if not db:
        print("⚠️ Neo4j not configured. Add credentials to .env")
        return
    
    try:
        # Get graph statistics
        stats = db.get_graph_stats()
        
        print(f"\n✓ Connected to Neo4j")
        print(f"  Total documents: {stats['total_nodes']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Label distribution: {stats['label_distribution']}")
        
    finally:
        db.close()


def example_neo4j_with_firerisk():
    """
    Example 2: Load FireRisk data and store in Neo4j
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: FireRisk Data to Neo4j")
    print("=" * 60)
    
    # Connect to Neo4j
    db = setup_neo4j_from_env()
    
    if not db:
        print("⚠️ Neo4j not configured")
        return
    
    try:
        # Option 1: Clear existing data (WARNING: destructive!)
        print("\nClearing previous data...")
        db.clear_all()
        
        # Load FireRisk dataset
        print("Loading FireRisk dataset...")
        loader = FireRiskLoader()
        docs = loader.download(limit=100)  # Use 100 for quick demo
        
        print(f"Loaded {len(docs)} documents")
        
        # Create graph edges
        print("Creating graph edges...")
        edges = create_graph_edges_from_documents(docs, similarity_threshold=0.7)
        print(f"Created {len(edges)} relationships")
        
        # Store in Neo4j
        print("Storing documents in Neo4j...")
        result = create_documents_with_neo4j(docs, edges, db)
        
        print(f"\n✓ Storage complete!")
        print(f"  Documents created: {result['documents_created']}")
        print(f"  Relationships created: {result['relationships_created']}")
        print(f"  Total nodes: {result['graph_stats']['total_nodes']}")
        print(f"  Total edges: {result['graph_stats']['total_relationships']}")
        
    finally:
        db.close()


def example_neo4j_query():
    """
    Example 3: Query documents from Neo4j
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Querying Neo4j")
    print("=" * 60)
    
    db = setup_neo4j_from_env()
    
    if not db:
        print("⚠️ Neo4j not configured")
        return
    
    try:
        # First, make sure there's data
        stats = db.get_graph_stats()
        if stats['total_nodes'] == 0:
            print("⚠️ No documents in Neo4j. Run example_neo4j_with_firerisk() first.")
            return
        
        print(f"\nGraph contains {stats['total_nodes']} documents\n")
        
        # Get first document ID (for demonstration)
        # In real usage, you'd know which document to query
        
        print("Sample document queries:")
        print("-" * 40)
        
        # Since we don't know document IDs, we'll show the query structure
        print("""
        # Find documents related to 'doc1'
        related = db.query_similar_documents('doc1', limit=5)
        
        # Find documents with same class as 'doc1'
        similar = db.query_similar_documents('doc1', relation_type='SAME_CLASS', limit=10)
        
        # Get full graph statistics
        stats = db.get_graph_stats()
        """)
        
        # Show actual graph statistics
        print("\nCurrent graph statistics:")
        print(f"  Total documents: {stats['total_nodes']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Class distribution: {stats['label_distribution']}")
        
    finally:
        db.close()


def example_neo4j_with_pipeline():
    """
    Example 4: Integrate Neo4j with multi-agent fine-tuning pipeline
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Neo4j + Multi-Agent Pipeline")
    print("=" * 60)
    
    # Setup Neo4j
    db = setup_neo4j_from_env()
    
    if not db:
        print("⚠️ Neo4j not configured")
        return
    
    try:
        # Create sample dataset
        print("\nPreparing dataset...")
        setup = create_firetask_setup()
        
        docs = setup['train_docs'] + setup['test_docs']
        edges = setup['edges']
        
        print(f"Dataset: {len(docs)} documents, {len(edges)} edges")
        
        # Store in Neo4j
        print("Storing in Neo4j...")
        result = create_documents_with_neo4j(docs, edges, db)
        print(f"✓ Stored {result['documents_created']} documents")
        
        # Get configuration
        config = get_default_config()
        
        # Run the multi-agent pipeline
        print("\nStarting multi-agent pipeline...")
        print("-" * 40)
        
        pipeline_results = run_multi_agent_pipeline(
            documents=docs,
            edges=edges,
            config=config
        )
        
        print("\n✓ Pipeline complete!")
        print(f"  Training status: {pipeline_results.get('status')}")
        
        # Check Neo4j stats after training
        stats = db.get_graph_stats()
        print(f"\nFinal Neo4j stats:")
        print(f"  Documents: {stats['total_nodes']}")
        print(f"  Relationships: {stats['total_relationships']}")
        
        # Save results
        results_file = "agent_outputs/neo4j_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'neo4j_stats': stats,
                'pipeline_results': pipeline_results
            }, f, indent=2)
        
        print(f"  Results saved to: {results_file}")
        
    finally:
        db.close()


def example_custom_neo4j_query():
    """
    Example 5: Run custom Cypher queries on Neo4j
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Cypher Queries")
    print("=" * 60)
    
    db = setup_neo4j_from_env()
    
    if not db:
        print("⚠️ Neo4j not configured")
        return
    
    try:
        stats = db.get_graph_stats()
        if stats['total_nodes'] == 0:
            print("⚠️ No documents in Neo4j")
            return
        
        print("\nRunning custom Cypher queries...\n")
        
        # Query 1: Find most connected documents
        with db.driver.session(database=db.database) as session:
            print("Most connected documents:")
            print("-" * 40)
            
            result = session.run("""
                MATCH (d:Document)-[r]-(other:Document)
                RETURN d.id as id, d.label as label, count(r) as degree
                ORDER BY degree DESC
                LIMIT 5
            """)
            
            for i, record in enumerate(result, 1):
                print(f"{i}. Doc {record['id']} (Label {record['label']}): {record['degree']} connections")
        
        # Query 2: Documents by class
        with db.driver.session(database=db.database) as session:
            print("\n\nDocuments by class:")
            print("-" * 40)
            
            result = session.run("""
                MATCH (d:Document)
                RETURN d.label as label, count(*) as count
                ORDER BY label
            """)
            
            for record in result:
                print(f"Class {record['label']}: {record['count']} documents")
        
    finally:
        db.close()


def main():
    """
    Run all examples
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Neo4j Integration Examples for GraphRAG Fine-Tuning  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run examples
    example_neo4j_basic()
    example_neo4j_with_firerisk()
    example_neo4j_query()
    example_custom_neo4j_query()
    example_neo4j_with_pipeline()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
    print("""
NEXT STEPS:
1. Review the NEO4J_SETUP_GUIDE.md for detailed documentation
2. Customize the integration for your use case
3. Use Neo4j Browser to visualize your graphs:
   https://console.neo4j.io/
4. Integrate with GraphRAG workflows
    """)


if __name__ == "__main__":
    main()
