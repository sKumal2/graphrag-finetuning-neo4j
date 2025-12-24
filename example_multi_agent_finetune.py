"""
Example: Multi-Agent Fine-Tuning Pipeline for GraphRAG
Demonstrates the supervisor agent orchestration pattern applied to graphRAG
"""

from fine_tune import run_multi_agent_pipeline, get_default_config


def prepare_sample_data():
    """Prepare sample documents and labels"""
    
    sample_documents = [
        {
            "id": "doc_1",
            "content": "Machine learning models require careful data preparation and validation",
            "metadata": {"source": "ml_handbook", "year": 2023}
        },
        {
            "id": "doc_2",
            "content": "Graph neural networks excel at capturing relational patterns",
            "metadata": {"source": "research_paper", "year": 2024}
        },
        {
            "id": "doc_3",
            "content": "Fine-tuning pre-trained embeddings improves downstream task performance",
            "metadata": {"source": "technique_guide", "year": 2024}
        },
        {
            "id": "doc_4",
            "content": "Vector databases enable efficient semantic search",
            "metadata": {"source": "db_docs", "year": 2023}
        },
        {
            "id": "doc_5",
            "content": "RAG systems combine retrieval with generation for knowledge-grounded outputs",
            "metadata": {"source": "rag_framework", "year": 2024}
        },
        {
            "id": "doc_6",
            "content": "Supervision signals guide model learning towards task objectives",
            "metadata": {"source": "training_guide", "year": 2023}
        },
        {
            "id": "doc_7",
            "content": "Multi-agent systems coordinate specialized models for complex workflows",
            "metadata": {"source": "orchestration_guide", "year": 2024}
        },
        {
            "id": "doc_8",
            "content": "Evaluation metrics help assess model quality objectively",
            "metadata": {"source": "metrics_guide", "year": 2023}
        },
    ]
    
    # Binary classification: 0 = Technical, 1 = Conceptual
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    
    return sample_documents, labels


def prepare_graph_edges():
    """Prepare knowledge graph edges for retriever"""
    
    edges = [
        ("doc_1", "doc_2"),  # Data prep → GNNs
        ("doc_2", "doc_3"),  # GNNs → Fine-tuning
        ("doc_3", "doc_5"),  # Fine-tuning → RAG
        ("doc_4", "doc_5"),  # Vector DB → RAG
        ("doc_5", "doc_7"),  # RAG → Multi-agent
        ("doc_6", "doc_7"),  # Supervision → Multi-agent
        ("doc_7", "doc_8"),  # Multi-agent → Evaluation
    ]
    
    return edges


def main():
    """Run the multi-agent fine-tuning pipeline"""
    
    print("\n" + "="*70)
    print("GRAPHRAG MULTI-AGENT FINE-TUNING EXAMPLE")
    print("="*70)
    
    # Step 1: Prepare data
    print("\n[1] Preparing sample data...")
    documents, labels = prepare_sample_data()
    edges = prepare_graph_edges()
    
    print(f"  • {len(documents)} documents loaded")
    print(f"  • {len(set(labels))} classes")
    print(f"  • {len(edges)} graph edges")
    
    # Step 2: Configure pipeline
    print("\n[2] Configuring pipeline...")
    config = get_default_config()
    
    # Optional: Override defaults
    config['epochs'] = 5  # Faster training for demo
    config['batch_size'] = 2
    config['output_dir'] = 'demo_outputs'
    
    print(f"  • Epochs: {config['epochs']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Output dir: {config['output_dir']}")
    
    # Step 3: Run pipeline
    print("\n[3] Running multi-agent pipeline...")
    print("    This will execute 5 coordinated agents:")
    print("    1. DataPreprationAgent     - Split & balance data")
    print("    2. RetrieverConfigAgent    - Setup graph retriever")
    print("    3. TrainingAgent           - Train embedding classifier")
    print("    4. EvaluationAgent         - Evaluate on test set")
    print("    5. ReportingAgent          - Generate visualizations")
    
    result = run_multi_agent_pipeline(
        documents=documents,
        labels=labels,
        edges=edges,
        config=config
    )
    
    # Step 4: Examine results
    print("\n[4] Results summary:")
    print(f"  • Data splits: {result['data']['train_labels'].__len__()} train, "
          f"{result['data']['val_labels'].__len__()} val, "
          f"{result['data']['test_labels'].__len__()} test")
    
    print(f"  • Best F1 (training): {result['training']['best_f1']:.4f}")
    print(f"  • Test F1: {result['evaluation']['test_f1']:.4f}")
    print(f"  • Test Accuracy: {result['evaluation']['test_acc']:.3%}")
    
    print(f"\n  • Artifacts saved to: {config['output_dir']}/")
    print(f"    - training_report.png")
    print(f"    - best_model.pth")
    print(f"    - Agent results (JSON)")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = main()
