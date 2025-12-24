# GraphRAG Multi-Agent Fine-Tuning Pipeline

## Overview

This project implements a **supervisor agent orchestration pattern** (inspired by [kshitizregmi/mootboard](https://github.com/kshitizregmi/mootboard)) for fine-tuning graphRAG embeddings and retriever models. The system uses 5 specialized agents coordinated by a supervisor to handle complex fine-tuning workflows.

## Architecture

```
┌─────────────────────────────────────────────────┐
│         SUPERVISOR AGENT (Orchestrator)         │
│  - Manages workflow execution                   │
│  - Coordinates agent communication              │
│  - Aggregates results                           │
└────────┬────────────────────────────────────────┘
         │
    ┌────┴───────────────────────────────────────┐
    ↓                                             ↓
┌──────────────────┐  ┌──────────────────────┐  ...
│ DataPrep Agent   │  │ RetrieverConfig Ag.  │
│ • Validate       │  │ • Setup vectorstore  │
│ • Split data     │  │ • Config graph edges │
│ • Balance classes│  │ • Initialize retriever
└────────┬─────────┘  └──────────────────────┘
         │                  
    ┌────┴──────────────────────────────────────┐
    ↓                                            ↓
┌──────────────────┐  ┌──────────────────────┐  ...
│ Training Agent   │  │ Evaluation Agent     │
│ • Train epoch    │  │ • Test inference     │
│ • Checkpoints    │  │ • Compute metrics    │
│ • Learning curves│  │ • Generate report    │
└──────────────────┘  └──────────────────────┘
         │                    
         └────────────┬───────────────────────┐
                      ↓
            ┌──────────────────────┐
            │ Reporting Agent      │
            │ • Visualizations     │
            │ • Summary statistics │
            │ • Artifact export    │
            └──────────────────────┘
```

## Agents

### 1. **DataPreprationAgent**
- **Responsibility**: Data validation and splitting
- **Input**: Raw documents & labels
- **Output**: Train/val/test splits, class distribution
- **Key Methods**: `execute(documents, labels)`

```python
agent = DataPreprationAgent("data_prep", config)
result = agent.execute(documents, labels)
# result['train_docs'], result['val_docs'], result['test_docs']
```

### 2. **RetrieverConfigAgent**
- **Responsibility**: Graph retriever setup
- **Input**: Graph edges, configuration
- **Output**: Initialized embeddings, vectorstore, retriever
- **Key Methods**: `execute(edges, config)`

```python
agent = RetrieverConfigAgent("retriever_config", config)
result = agent.execute(edges, config)
# result['retriever'], result['embeddings'], result['vectorstore']
```

### 3. **TrainingAgent**
- **Responsibility**: Model training loop
- **Input**: Data loaders, training config
- **Output**: Trained model, training history
- **Key Methods**: `execute(train_loader, val_loader, ...)`
- **Features**:
  - Weighted sampling for imbalanced data
  - Cosine annealing learning rate schedule
  - Gradient clipping
  - Checkpoint management

```python
agent = TrainingAgent("training", config)
result = agent.execute(train_loader, val_loader, num_classes, class_weights, config)
# result['model_state'], result['history'], result['best_f1']
```

### 4. **EvaluationAgent**
- **Responsibility**: Test set evaluation
- **Input**: Model state, test loader
- **Output**: Metrics (F1, accuracy, precision, recall)
- **Key Methods**: `execute(model_state, test_loader, ...)`
- **Metrics**: F1, accuracy, precision, recall

```python
agent = EvaluationAgent("evaluation", config)
result = agent.execute(model_state, test_loader, num_classes, config)
# result['test_f1'], result['test_acc'], result['predictions']
```

### 5. **ReportingAgent**
- **Responsibility**: Visualization & reporting
- **Input**: Training history, evaluation metrics
- **Output**: Plots, summary statistics
- **Key Methods**: `execute(training_history, eval_metrics)`
- **Outputs**:
  - Training/validation loss curve
  - Accuracy curve
  - F1 score curve
  - Test metrics summary

```python
agent = ReportingAgent("reporting", config)
result = agent.execute(training_history, eval_metrics)
# result['report_file'] = 'training_report.png'
```

### **SupervisorAgent** (Orchestrator)
- **Responsibility**: Workflow orchestration
- **Method**: `orchestrate(documents, labels, edges)`
- **Workflow**:
  1. Register agents
  2. Execute pipeline in sequence
  3. Pass outputs between agents
  4. Aggregate final results
  5. Generate summary report

## Usage

### Basic Usage

```python
from fine_tune import run_multi_agent_pipeline

# Prepare your data
documents = [
    {"content": "Document 1 text", "metadata": {...}},
    {"content": "Document 2 text", "metadata": {...}},
    ...
]
labels = [0, 1, 0, 1, ...]  # Class labels

# Define graph edges (for retriever)
edges = [
    ("doc_1", "doc_2"),
    ("doc_2", "doc_3"),
    ...
]

# Run pipeline with default config
result = run_multi_agent_pipeline(
    documents=documents,
    labels=labels,
    edges=edges
)
```

### Advanced Usage

```python
from fine_tune import get_default_config, run_multi_agent_pipeline

# Customize configuration
config = get_default_config()
config['epochs'] = 50
config['batch_size'] = 16
config['learning_rate'] = 1e-4

# Run with custom config
result = run_multi_agent_pipeline(
    documents=documents,
    labels=labels,
    edges=edges,
    config=config
)

# Access results from each agent
print(result['data']['class_distribution'])
print(result['training']['best_f1'])
print(result['evaluation']['test_f1'])
print(result['report']['summary'])
```

### Using Individual Agents

```python
from multi_agent_orchestration import (
    DataPreprationAgent,
    RetrieverConfigAgent,
    TrainingAgent,
    EvaluationAgent,
    ReportingAgent
)

# Create and execute agents manually
prep_agent = DataPreprationAgent("data_prep", config)
data_result = prep_agent.execute(documents, labels)

retriever_agent = RetrieverConfigAgent("retriever_config", config)
retriever_result = retriever_agent.execute(edges, config)

# ... continue with other agents
```

## File Structure

```
.
├── fine_tune.py                          # Core fine-tuning module
├── multi_agent_orchestration.py          # Agent classes & supervisor
├── example_multi_agent_finetune.py       # Usage example
├── requirements.txt                      # Dependencies
├── .env                                  # Environment variables
└── agent_outputs/                        # Generated artifacts
    ├── data_prep_results.json
    ├── retriever_config_results.json
    ├── training_results.json
    ├── evaluation_results.json
    ├── reporting_results.json
    ├── training_report.png
    └── best_model.pth
```

## Configuration

Key configuration parameters:

```python
{
    'epochs': 30,                          # Training epochs
    'batch_size': 32,                      # Batch size
    'learning_rate': 3e-4,                 # Adam LR
    'weight_decay': 0.05,                  # L2 regularization
    'embedding_dim': 768,                  # Embedding dimension
    'embedding_model': 'models/embedding-001',  # Google Generative AI model
    'max_grad_norm': 1.0,                  # Gradient clipping
    'collection_name': 'finetuned_docs',   # Chroma collection
    'database_name': 'graphRAG_finetuned', # Chroma database
    'output_dir': 'agent_outputs',         # Output directory
}
```

## Training Details

### Class Weighting
- Automatically calculates weights for imbalanced datasets
- Uses inverse frequency weighting: `weight = total / (num_classes * count)`

### Learning Rate Schedule
- **Optimizer**: AdamW with weight decay
- **Schedule**: Cosine annealing with minimum LR of 1e-6
- **Formula**: `lr(t) = 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + cos(π * t / T_max))`

### Regularization
- **Gradient clipping**: max_norm=1.0
- **Weight decay**: 0.05
- **Loss**: Weighted cross-entropy

## Evaluation Metrics

- **F1 Score**: Both weighted and macro averages
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **AUC**: Area under ROC curve (for binary classification)

## Running the Example

```bash
python example_multi_agent_finetune.py
```

This will:
1. Create sample documents and labels
2. Setup graph edges
3. Execute the full multi-agent pipeline
4. Generate visualizations
5. Print performance metrics

## Integration with Existing Code

The new system is fully backward compatible with existing `fine_tune.py` code:

```python
# Old way (still works)
from fine_tune import EmbeddingFinetuner, create_data_loaders

finetuner = EmbeddingFinetuner(config)
train_loader, val_loader, test_loader, num_classes = create_data_loaders(...)
finetuner.train(train_loader, val_loader, test_loader, num_classes)

# New way (multi-agent)
from fine_tune import run_multi_agent_pipeline

result = run_multi_agent_pipeline(documents, labels, edges)
```

## Key Improvements

1. **Modularity**: Each agent is independent and testable
2. **Orchestration**: Clear workflow management via supervisor
3. **Extensibility**: Easy to add new agents (e.g., DataAugmentationAgent, ExplanationAgent)
4. **Observability**: Each agent saves results to disk for inspection
5. **Scalability**: Agents can be parallelized or run on separate workers
6. **Error Handling**: Each agent has status tracking and error logging

## Future Enhancements

1. **Async Agents**: Parallel execution using asyncio (like mootboard's video generation polling)
2. **Caching**: Agent output caching to avoid re-computation
3. **Monitoring**: Real-time progress tracking and metrics visualization
4. **Distributed**: Multi-worker deployment using Ray or similar
5. **Data Augmentation Agent**: Synthetic data generation for imbalanced classes
6. **Hyperparameter Tuning Agent**: Automated HPO using Optuna or Ray Tune

## References

- **Mootboard**: [kshitizregmi/mootboard](https://github.com/kshitizregmi/mootboard) - Multi-agent orchestration pattern
- **skinLesion_VIT**: Reference project for fine-tuning approach
- **LangChain**: Embedding & retrieval framework
- **Chroma**: Vector database
- **Google Generative AI**: Embedding model

## License

Same as main graphRAG project
