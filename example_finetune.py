"""
Example usage of the fine-tuning module for graphRAG
Shows how to prepare data and fine-tune the embedding model
"""

import torch
from pathlib import Path
from fine_tune import (
    FineTuneConfig,
    EmbeddingFinetuner,
    DocumentDataset,
    create_data_loaders
)


def prepare_sample_data():
    """
    Prepare sample training data
    Replace this with your actual data loading logic
    """
    
    # Example: Movie documents with labels (for your graphRAG project)
    train_docs = [
        {
            'content': 'The Shawshank Redemption is a 1994 drama film',
            'metadata': {'year': 1994, 'genre': 'drama'}
        },
        {
            'content': 'Pulp Fiction was released in 1994',
            'metadata': {'year': 1994, 'genre': 'crime'}
        },
        {
            'content': 'Forrest Gump came out in 1994',
            'metadata': {'year': 1994, 'genre': 'drama'}
        },
        # Add more documents...
    ]
    
    train_labels = [0, 1, 0]  # Example: 0=drama, 1=crime
    
    val_docs = [
        {
            'content': 'The Lion King is a 1994 animated film',
            'metadata': {'year': 1994, 'genre': 'animation'}
        },
    ]
    val_labels = [2]
    
    test_docs = [
        {
            'content': 'Toy Story came out in 1995',
            'metadata': {'year': 1995, 'genre': 'animation'}
        },
    ]
    test_labels = [2]
    
    return train_docs, train_labels, val_docs, val_labels, test_docs, test_labels


def main():
    """
    Complete fine-tuning pipeline
    """
    
    print("="*90)
    print("GraphRAG Embedding Fine-Tuning Pipeline")
    print("="*90)
    
    # Initialize config
    config = FineTuneConfig()
    print(f"\nConfig:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Model: {config.MODEL_NAME}")
    
    # Prepare data
    print("\n[1] Preparing data...")
    train_docs, train_labels, val_docs, val_labels, test_docs, test_labels = prepare_sample_data()
    
    print(f"  Train samples: {len(train_docs)}")
    print(f"  Val samples: {len(val_docs)}")
    print(f"  Test samples: {len(test_docs)}")
    
    # Create data loaders
    print("\n[2] Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(
        train_docs, train_labels,
        val_docs, val_labels,
        test_docs, test_labels,
        batch_size=config.BATCH_SIZE
    )
    print(f"  Number of classes: {num_classes}")
    
    # Initialize fine-tuner
    print("\n[3] Initializing fine-tuner...")
    finetuner = EmbeddingFinetuner(config)
    print(f"  Device: {finetuner.device}")
    
    # Setup retriever with graph edges
    print("\n[4] Setting up retriever...")
    movie_edges = [
        ("release_year", "release_year"),
        ("movie_genre", "movie_genre"),
    ]
    finetuner.setup_retriever(movie_edges)
    print(f"  Retriever configured with {len(movie_edges)} edges")
    
    # Get class weights for imbalanced data
    train_dataset = DocumentDataset(train_docs, train_labels)
    class_weights = train_dataset.get_class_weights()
    
    # Start fine-tuning
    print("\n[5] Starting fine-tuning...")
    finetuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        class_weights=class_weights
    )
    
    print("\n[6] Fine-tuning complete!")
    print(f"  Best model: {config.BEST_MODEL_PATH}")
    print(f"  Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"  Learning curves: learning_curves.png")
    
    # Return finetuner for further use
    return finetuner


if __name__ == "__main__":
    finetuner = main()
    
    # Example: Use the finetuned retriever
    # results = finetuner.retriever.invoke("what movies were released in 1994?")
