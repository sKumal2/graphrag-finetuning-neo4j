# GraphRAG Fine-Tuning: Complete Setup Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Run Setup Script
```bash
python finetune_setup.py
```
This will:
- ✓ Check and install dependencies
- ✓ Create project directories
- ✓ Download FireRisk dataset (~5000 samples)
- ✓ Generate startup scripts
- ✓ Create configuration file (.env)

### Step 2: Verify Installation
```bash
python test_setup.py
```
This validates:
- ✓ Dataset loading
- ✓ Data loaders
- ✓ Configuration
- ✓ Batch creation

### Step 3: Start Fine-Tuning
```bash
python start_finetuning.py
```
This executes:
- ✓ Multi-agent pipeline
- ✓ Training with checkpointing
- ✓ Evaluation metrics
- ✓ Visualization generation

---

## 📊 Dataset: FireRisk

**Source**: [HuggingFace - FireRisk](https://huggingface.co/datasets/blanchon/FireRisk)

### Dataset Details
| Property | Value |
|----------|-------|
| Total Images | 91,872 |
| Sample Size (Demo) | 5,000 |
| Image Size | 320×320 pixels |
| Bands | 3 (RGB) |
| Classes | 7 |
| Resolution | 1m |
| Source | NAIP Aerial Imagery |

### Classes
1. **high** (class 0)
2. **low** (class 1)
3. **moderate** (class 2)
4. **non-burnable** (class 3)
5. **very_high** (class 4)
6. **very_low** (class 5)
7. **water** (class 6)

### Data Format
Each sample converted to document format:
```python
{
    'id': 'firerisk_train_0',
    'content': 'Fire risk level: high. Remote sensing image analysis.',
    'label': 0,
    'label_name': 'high',
    'metadata': {
        'source': 'FireRisk',
        'split': 'train',
        'image_size': (320, 320),
        'bands': 3,
    }
}
```

---

## 🏗️ Project Structure

```
New folder/
├── Core Modules
│   ├── fine_tune.py                      # Fine-tuning trainer
│   ├── multi_agent_orchestration.py      # Agent classes
│   ├── data_loaders.py                   # Dataset utilities
│   └── requirements.txt                  # Dependencies
│
├── Setup & Startup
│   ├── finetune_setup.py                 # Setup script
│   ├── start_finetuning.py              # Quick start
│   ├── test_setup.py                    # Verification
│   └── .env                             # Configuration
│
├── Documentation
│   ├── SETUP_GUIDE.md                   # This file
│   ├── FINETUNING_README.md             # Usage guide
│   ├── MULTI_AGENT_ARCHITECTURE.md      # Architecture
│   └── example_multi_agent_finetune.py  # Example
│
└── Runtime Directories
    ├── data/
    │   └── firerisk/                     # Downloaded dataset
    ├── checkpoints/                      # Model checkpoints
    ├── agent_outputs/                    # Agent results
    ├── logs/                             # Training logs
    └── visualizations/                   # Generated plots
```

---

## ⚙️ Configuration

### Environment Variables (.env)

Create `.env` file with your API keys:

```env
# Google API (Required for embeddings)
GOOGLE_API_KEY=your_google_api_key

# Chroma Vector Database (Optional)
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=default
CHROMA_DATABASE=graphRAG_finetuned

# HuggingFace (Optional - for dataset downloads)
HUGGINGFACE_TOKEN=your_hf_token

# Experiment Tracking (Optional)
WANDB_API_KEY=your_wandb_key
```

### Training Configuration

Edit `start_finetuning.py` to customize:

```python
config = get_default_config()

# Training parameters
config['epochs'] = 30                  # Number of training epochs
config['batch_size'] = 32              # Batch size
config['learning_rate'] = 3e-4        # Adam learning rate
config['weight_decay'] = 0.05         # L2 regularization

# Model parameters
config['embedding_dim'] = 768          # Embedding dimension
config['embedding_model'] = 'models/embedding-001'  # Google Generative AI model
config['max_grad_norm'] = 1.0         # Gradient clipping

# Storage
config['output_dir'] = 'firerisk_outputs'  # Output directory
```

---

## 🧠 Multi-Agent Architecture

### Pipeline Overview

```
INPUT: Documents + Labels + Graph Edges
  ↓
[SUPERVISOR AGENT] (Orchestrator)
  ├─→ DataPrepAgent: Split & balance data
  │   └─→ Output: train/val/test splits
  │
  ├─→ RetrieverConfigAgent: Setup retriever
  │   └─→ Output: Embeddings, vectorstore, retriever
  │
  ├─→ TrainingAgent: Train classifier
  │   ├─ Weighted sampling for class imbalance
  │   ├─ Cosine annealing LR schedule
  │   ├─ Checkpoint management
  │   └─→ Output: Trained model, history
  │
  ├─→ EvaluationAgent: Test on held-out set
  │   ├─ Compute F1, accuracy, precision, recall
  │   └─→ Output: Metrics & predictions
  │
  └─→ ReportingAgent: Generate visualizations
      ├─ Loss curves (train/val)
      ├─ Accuracy curves
      ├─ F1 score curves
      └─→ Output: training_report.png
  
OUTPUT: Model + Metrics + Visualizations
```

### Agents

#### 1. DataPreprationAgent
- **Purpose**: Validate and split data
- **Input**: Raw documents & labels
- **Output**: Stratified train/val/test splits
- **Features**:
  - Class distribution analysis
  - Stratified splitting
  - Imbalance detection

#### 2. RetrieverConfigAgent
- **Purpose**: Initialize retrieval system
- **Input**: Graph edges, configuration
- **Output**: Embeddings, vectorstore, retriever
- **Models**:
  - Embeddings: Google Generative AI (768-dim)
  - Vector DB: Chroma
  - Retriever: GraphRetriever with EagerStrategy

#### 3. TrainingAgent
- **Purpose**: Train classification head
- **Input**: Data loaders, hyperparameters
- **Output**: Trained model, training history
- **Features**:
  - Weighted cross-entropy loss
  - AdamW optimizer with weight decay
  - Cosine annealing scheduler
  - Gradient clipping (max_norm=1.0)
  - Checkpoint management

#### 4. EvaluationAgent
- **Purpose**: Evaluate model performance
- **Input**: Model state, test data
- **Output**: Comprehensive metrics
- **Metrics**:
  - Weighted F1 score
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - AUC (for binary)

#### 5. ReportingAgent
- **Purpose**: Generate visualizations
- **Input**: Training history, evaluation metrics
- **Output**: Plots and summaries
- **Visualizations**:
  - Loss curve (train/val)
  - Accuracy curve (train/val)
  - F1 score curve
  - Test metrics summary

---

## 📦 Installation Details

### Dependencies

```
torch>=2.0.0                    # Deep learning framework
langchain>=0.1.0                # LLM/embedding framework
langchain-google-genai          # Google Generative AI integration
langchain-chroma                # Vector database integration
chromadb>=0.4.0                 # Vector database
scikit-learn>=1.3.0             # ML utilities & metrics
matplotlib>=3.7.0               # Visualization
datasets>=2.14.0                # HuggingFace datasets
huggingface-hub>=0.17.0         # HuggingFace utilities
transformers>=4.30.0            # Pre-trained models
python-dotenv>=1.0.0            # Environment variables
tqdm                            # Progress bars
numpy                           # Numerical computing
```

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install torch langchain langchain-google-genai chromadb scikit-learn
```

---

## 🚀 Usage Examples

### Basic Usage

```python
from fine_tune import run_multi_agent_pipeline
from data_loaders import FireRiskLoader, convert_to_document_format

# Load FireRisk dataset
loader = FireRiskLoader()
samples = loader.download(split="train", limit=5000)

# Convert to document format
documents, labels = convert_to_document_format(samples)

# Create graph edges
from data_loaders import create_graph_edges_from_documents
edges = create_graph_edges_from_documents(documents, edge_strategy="sequential")

# Run pipeline
result = run_multi_agent_pipeline(
    documents=documents,
    labels=labels,
    edges=edges
)

# Access results
print(f"Best F1: {result['training']['best_f1']:.4f}")
print(f"Test Accuracy: {result['evaluation']['test_acc']:.3%}")
```

### Advanced Usage with Custom Config

```python
from fine_tune import get_default_config, run_multi_agent_pipeline

# Customize configuration
config = get_default_config()
config['epochs'] = 50
config['batch_size'] = 16
config['learning_rate'] = 1e-4
config['output_dir'] = 'custom_outputs'

# Run with custom config
result = run_multi_agent_pipeline(
    documents=documents,
    labels=labels,
    edges=edges,
    config=config
)
```

### Using Individual Agents

```python
from multi_agent_orchestration import (
    DataPreprationAgent,
    TrainingAgent,
    EvaluationAgent,
)

# Create agents
prep_agent = DataPreprationAgent("data_prep", config)
data_result = prep_agent.execute(documents, labels)

train_agent = TrainingAgent("training", config)
train_result = train_agent.execute(train_loader, val_loader, num_classes, class_weights, config)

eval_agent = EvaluationAgent("evaluation", config)
eval_result = eval_agent.execute(train_result['model_state'], test_loader, num_classes, config)
```

---

## 📊 Training Details

### Learning Rate Schedule

Uses cosine annealing with warm restarts:

$$\text{lr}(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi \cdot \frac{t}{T_{\max}}))$$

Where:
- $\eta_0 = 3 \times 10^{-4}$ (initial learning rate)
- $\eta_{\min} = 1 \times 10^{-6}$ (minimum learning rate)
- $T_{\max} = 30$ (number of epochs)

### Class Weighting

For imbalanced datasets, weights are calculated as:

$$w_c = \frac{N}{n_c \times C}$$

Where:
- $N$ = total number of samples
- $n_c$ = number of samples in class $c$
- $C$ = number of classes

### Loss Function

Weighted cross-entropy loss:

$$\mathcal{L} = -\sum_{c=1}^{C} w_c \times \text{CE}(y_c, \hat{y}_c)$$

### Regularization

- **Gradient clipping**: $||\nabla|| \leq 1.0$
- **Weight decay**: $\lambda = 0.05$
- **Optimizer**: AdamW with $\beta_1=0.9, \beta_2=0.999$

---

## 📈 Output & Results

### Generated Files

After training, you'll have:

```
agent_outputs/
├── data_prep_results.json          # Data split info
├── retriever_config_results.json   # Retriever setup
├── training_results.json           # Training history
├── evaluation_results.json         # Test metrics
├── reporting_results.json          # Report metadata
└── training_report.png             # 4-panel visualization

checkpoints/
├── best_model_f1.pth              # Best model
├── epoch_10_ckpt.pth              # Periodic checkpoints
└── best_model_f1.optimizer.pth    # Optimizer state

logs/
└── training.log                    # Detailed logs
```

### Visualization

The `training_report.png` contains 4 subplots:

1. **Loss Curves** (top-left)
   - Training loss (blue)
   - Validation loss (orange)

2. **Accuracy Curves** (top-right)
   - Training accuracy (blue)
   - Validation accuracy (orange)

3. **F1 Score** (bottom-left)
   - Validation F1 (weighted)

4. **Test Metrics** (bottom-right)
   - Text summary of test performance

---

## 🔧 Troubleshooting

### Problem: API Key Error

**Error**: `Missing GOOGLE_API_KEY`

**Solution**:
1. Get API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Add to `.env`:
   ```env
   GOOGLE_API_KEY=your_key_here
   ```
3. Restart script

### Problem: Out of Memory

**Error**: `CUDA out of memory`

**Solution**: Reduce batch size
```python
config['batch_size'] = 8  # Instead of 32
```

### Problem: Dataset Download Fails

**Error**: Network/authentication issues

**Solution**: Script automatically uses mock data
```python
# In finetune_setup.py
setup_data = create_mock_dataset()
```

### Problem: Slow Training

**Optimization**:
- Reduce `epochs` for quick test
- Use smaller `batch_size` (trades speed for memory)
- Limit dataset with `limit=1000` parameter

---

## 📚 Next Steps

### 1. Integrate with GraphRAG
```python
from fine_tune import EmbeddingFinetuner
from langchain_graph_retriever import GraphRetriever

# Use fine-tuned embeddings in GraphRAG
finetuner = EmbeddingFinetuner(config)
finetuner.setup_retriever(edges)
retriever = finetuner.retriever
```

### 2. Deploy Model
```python
# Save for production
torch.save(model.state_dict(), "model.pth")

# Load in production
model = load_model("model.pth")
```

### 3. Monitor Training
Integrate with experiment tracking:
```python
# WandB
import wandb
wandb.init(project="graphrag-finetuning")

# MLflow
import mlflow
mlflow.start_run()
```

### 4. Hyperparameter Optimization
```python
from optuna import create_study

def objective(trial):
    config['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-3)
    config['batch_size'] = trial.suggest_int('batch_size', 8, 64)
    result = run_multi_agent_pipeline(documents, labels, edges, config)
    return result['evaluation']['test_f1']

study = create_study()
study.optimize(objective, n_trials=20)
```

---

## 📖 Additional Resources

- **LangChain Docs**: [python.langchain.com](https://python.langchain.com)
- **Chroma Docs**: [docs.trychroma.com](https://docs.trychroma.com)
- **PyTorch**: [pytorch.org](https://pytorch.org)
- **scikit-learn**: [scikit-learn.org](https://scikit-learn.org)
- **FireRisk Paper**: [arxiv.org/abs/2303.07035](https://arxiv.org/abs/2303.07035)

---

## 📞 Support

For issues or questions:
1. Check `test_setup.py` output
2. Review logs in `logs/` directory
3. Check `.env` configuration
4. Review agent output in `agent_outputs/`

---

**Happy Fine-Tuning! 🚀**
