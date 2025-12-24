"""
Complete Fine-Tuning Environment Setup
Downloads FireRisk dataset and configures multi-agent pipeline
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check and install required dependencies"""
    logger.info("="*70)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*70)
    
    required_packages = {
        'torch': 'torch',
        'datasets': 'datasets',
        'langchain': 'langchain',
        'langchain_google_genai': 'langchain-google-genai',
        'chromadb': 'chromadb',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'dotenv': 'python-dotenv',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        logger.info(f"\nInstalling {len(missing)} missing packages...")
        import subprocess
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        logger.info("✓ All packages installed")
    
    return len(missing) == 0


def setup_environment_variables():
    """Setup .env file with required API keys"""
    logger.info("\n" + "="*70)
    logger.info("ENVIRONMENT SETUP")
    logger.info("="*70)
    
    env_file = Path(".env")
    
    # Check if .env exists
    if env_file.exists():
        logger.info("✓ .env file already exists")
        return
    
    # Create template
    logger.info("Creating .env template...")
    env_template = """# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Chroma Vector Database
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=default
CHROMA_DATABASE=graphRAG_finetuned

# Optional: HuggingFace Configuration
HUGGINGFACE_TOKEN=your_hf_token_here

# Optional: Weights & Biases (for experiment tracking)
WANDB_API_KEY=your_wandb_key_here
"""
    
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    logger.info(f"✓ Created {env_file}")
    logger.info("\n⚠️  Please update .env with your API keys!")


def create_project_structure():
    """Create project directory structure"""
    logger.info("\n" + "="*70)
    logger.info("CREATING PROJECT STRUCTURE")
    logger.info("="*70)
    
    directories = [
        Path("data"),
        Path("data/firerisk"),
        Path("checkpoints"),
        Path("agent_outputs"),
        Path("logs"),
        Path("visualizations"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ {directory}")


def load_firerisk_dataset() -> Dict:
    """Load FireRisk dataset"""
    logger.info("\n" + "="*70)
    logger.info("LOADING FIRERISK DATASET")
    logger.info("="*70)
    
    from data_loaders import create_firetask_setup
    
    try:
        setup_data = create_firetask_setup()
        return setup_data
    except Exception as e:
        logger.error(f"Error loading FireRisk: {e}")
        logger.info("Creating mock dataset instead...")
        return create_mock_dataset()


def create_mock_dataset() -> Dict:
    """Create mock dataset for testing"""
    logger.info("\nCreating mock dataset for testing...")
    
    mock_docs = [
        {
            "id": f"doc_{i}",
            "content": f"Sample fire risk assessment document {i}. "
                      f"This document discusses risk level: {['high', 'low', 'moderate'][i % 3]}.",
            "metadata": {"source": "mock", "index": i}
        }
        for i in range(100)
    ]
    
    mock_labels = [i % 3 for i in range(100)]
    
    mock_edges = [
        (f"doc_{i}", f"doc_{i+1}")
        for i in range(len(mock_docs) - 1)
    ]
    
    config = {
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'embedding_dim': 768,
        'embedding_model': 'models/embedding-001',
        'output_dir': 'mock_outputs',
        'dataset_name': 'Mock',
        'num_classes': 3,
    }
    
    logger.info(f"✓ Created mock dataset with {len(mock_docs)} documents")
    
    return {
        'train_docs': mock_docs,
        'train_labels': mock_labels,
        'test_docs': mock_docs[:20],
        'test_labels': mock_labels[:20],
        'edges': mock_edges,
        'config': config,
        'label_names': {0: 'high', 1: 'low', 2: 'moderate'},
    }


def create_startup_script(setup_data: Dict):
    """Create a startup script for quick training"""
    logger.info("\n" + "="*70)
    logger.info("CREATING STARTUP SCRIPTS")
    logger.info("="*70)
    
    startup_script = '''"""
Quick Start: Begin Fine-Tuning with FireRisk Dataset
Run this script to start the full multi-agent pipeline
"""

import logging
from data_loaders import create_firetask_setup
from fine_tune import run_multi_agent_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("\\n" + "="*70)
    logger.info("STARTING FIRERISK FINE-TUNING")
    logger.info("="*70)
    
    # Load dataset
    logger.info("\\n[1] Loading FireRisk dataset...")
    try:
        setup_data = create_firetask_setup()
    except Exception as e:
        logger.error(f"Failed to load FireRisk: {e}")
        logger.info("Using mock dataset instead...")
        from finetune_setup import create_mock_dataset
        setup_data = create_mock_dataset()
    
    # Extract data
    train_docs = setup_data['train_docs']
    train_labels = setup_data['train_labels']
    edges = setup_data['edges']
    config = setup_data['config']
    
    logger.info(f"  • Dataset: {config['dataset_name']}")
    logger.info(f"  • Train samples: {len(train_docs)}")
    logger.info(f"  • Classes: {config['num_classes']}")
    
    # Run pipeline
    logger.info("\\n[2] Starting multi-agent fine-tuning pipeline...")
    logger.info("    Agents executing in sequence:")
    logger.info("    1. DataPrepPrep")
    logger.info("    2. RetrieverConfig")
    logger.info("    3. Training")
    logger.info("    4. Evaluation")
    logger.info("    5. Reporting")
    
    result = run_multi_agent_pipeline(
        documents=train_docs,
        labels=train_labels,
        edges=edges,
        config=config
    )
    
    # Summary
    logger.info("\\n" + "="*70)
    logger.info("✓ FINE-TUNING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best F1: {result['training']['best_f1']:.4f}")
    logger.info(f"Test F1: {result['evaluation']['test_f1']:.4f}")
    logger.info(f"Test Accuracy: {result['evaluation']['test_acc']:.3%}")
    logger.info(f"Output: {config['output_dir']}/")
'''
    
    with open("start_finetuning.py", 'w') as f:
        f.write(startup_script)
    
    logger.info("✓ Created start_finetuning.py")
    
    # Create quick test script
    test_script = '''"""
Quick Test: Verify setup works
Run this to test the environment before full training
"""

import logging
from fine_tune import get_default_config, create_data_loaders
from data_loaders import FireRiskLoader, convert_to_document_format

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("\\nVerifying fine-tuning setup...")
    
    # Test 1: Load config
    logger.info("\\n[1] Testing configuration...")
    config = get_default_config()
    logger.info(f"  ✓ Loaded config with {len(config)} parameters")
    
    # Test 2: Load dataset
    logger.info("\\n[2] Testing dataset loading...")
    try:
        loader = FireRiskLoader()
        samples = loader.download(split="train", limit=50)
        logger.info(f"  ✓ Loaded {len(samples)} samples")
    except:
        logger.warning("  ! FireRisk not available, using mock")
        from finetune_setup import create_mock_dataset
        setup = create_mock_dataset()
        samples = [
            {'content': d['content'], 'label': l}
            for d, l in zip(setup['train_docs'], setup['train_labels'])
        ]
    
    # Test 3: Create data loaders
    logger.info("\\n[3] Testing DataLoader creation...")
    docs, labels = convert_to_document_format(samples[:40])
    
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(
        train_docs=docs[:30],
        train_labels=labels[:30],
        val_docs=docs[30:35],
        val_labels=labels[30:35],
        test_docs=docs[35:40],
        test_labels=labels[35:40],
        batch_size=8
    )
    logger.info(f"  ✓ Created loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    logger.info(f"  ✓ Classes: {num_classes}")
    
    # Test 4: Sample batch
    logger.info("\\n[4] Testing batch loading...")
    batch = next(iter(train_loader))
    logger.info(f"  ✓ Batch loaded: {len(batch['text'])} samples")
    
    logger.info("\\n" + "="*70)
    logger.info("✓ ALL TESTS PASSED - Ready for fine-tuning!")
    logger.info("="*70)
    logger.info("\\nNext step: python start_finetuning.py")
'''
    
    with open("test_setup.py", 'w') as f:
        f.write(test_script)
    
    logger.info("✓ Created test_setup.py")


def create_readme():
    """Create README for quick reference"""
    logger.info("\n" + "="*70)
    logger.info("CREATING DOCUMENTATION")
    logger.info("="*70)
    
    readme = """# GraphRAG Fine-Tuning Environment

## Quick Start

### 1. Setup Environment
```bash
# Run once to setup everything
python finetune_setup.py
```

### 2. Verify Setup
```bash
# Test that everything works
python test_setup.py
```

### 3. Start Fine-Tuning
```bash
# Run the full multi-agent pipeline
python start_finetuning.py
```

## Project Structure

```
.
├── fine_tune.py                    # Core fine-tuning module
├── multi_agent_orchestration.py    # Agent classes & supervisor
├── data_loaders.py                 # Dataset loading utilities
├── example_multi_agent_finetune.py # Example usage
├── start_finetuning.py            # Quick start script
├── test_setup.py                  # Verification script
├── finetune_setup.py              # Setup script
├── requirements.txt                # Dependencies
├── .env                            # API keys (configure this!)
│
├── data/
│   └── firerisk/                   # FireRisk dataset
├── checkpoints/                    # Model checkpoints
├── agent_outputs/                  # Agent results
├── logs/                           # Training logs
└── visualizations/                 # Generated plots
```

## Configuration

Edit `.env` with your API keys:
```env
GOOGLE_API_KEY=your_key
CHROMA_API_KEY=your_key
HUGGINGFACE_TOKEN=your_token
```

## Dataset: FireRisk

- **Source**: Hugging Face (blanchon/FireRisk)
- **Size**: ~91,872 images (loaded as 5k for quick start)
- **Classes**: 7 (high, low, moderate, non-burnable, very_high, very_low, water)
- **Features**: Remote sensing fire risk classification
- **Format**: Converted to text documents for fine-tuning

## Training Pipeline

### Agents (Executed in Sequence)

1. **DataPrepAgent**
   - Validates data
   - Splits into train/val/test
   - Calculates class weights

2. **RetrieverConfigAgent**
   - Initializes embeddings (Google Generative AI)
   - Sets up vector database (Chroma)
   - Configures graph retriever

3. **TrainingAgent**
   - Trains embedding classifier
   - Applies class weighting for imbalance
   - Checkpoints best models

4. **EvaluationAgent**
   - Tests on held-out test set
   - Computes metrics (F1, accuracy, precision, recall)
   - Generates predictions

5. **ReportingAgent**
   - Creates visualizations (loss, accuracy, F1 curves)
   - Generates summary statistics
   - Exports metrics

## Customization

Edit `start_finetuning.py` to adjust:

```python
config = get_default_config()
config['epochs'] = 30          # Training epochs
config['batch_size'] = 32      # Batch size
config['learning_rate'] = 3e-4 # Learning rate
```

## Output Artifacts

After training:
- `best_model.pth` - Best trained model weights
- `training_report.png` - 4-panel visualization
- `agent_outputs/` - JSON results from each agent
- `logs/` - Training logs

## Troubleshooting

### API Keys Not Set
```python
# Add to .env
GOOGLE_API_KEY=your_key
CHROMA_API_KEY=your_key
```

### Dataset Download Fails
The script automatically falls back to mock data for testing.

### Out of Memory
Reduce `batch_size` in config:
```python
config['batch_size'] = 8  # Instead of 32
```

## Next Steps

1. Integrate with graphRAG retrieval system
2. Fine-tune on domain-specific documents
3. Deploy model for production inference
4. Monitor with experiment tracking (W&B, MLflow)

## References

- **Framework**: LangChain + PyTorch
- **Dataset**: FireRisk (remote sensing, fire risk)
- **Architecture**: Multi-agent orchestration (inspired by mootboard)
- **Embeddings**: Google Generative AI
- **Vector DB**: Chroma
"""
    
    with open("FINETUNING_README.md", 'w') as f:
        f.write(readme)
    
    logger.info("✓ Created FINETUNING_README.md")


def print_summary(setup_data: Dict):
    """Print setup summary"""
    logger.info("\n" + "="*70)
    logger.info("SETUP COMPLETE ✓")
    logger.info("="*70)
    
    config = setup_data['config']
    
    logger.info("\nDataset Configuration:")
    logger.info(f"  • Name: {config['dataset_name']}")
    logger.info(f"  • Train samples: {len(setup_data['train_docs'])}")
    logger.info(f"  • Test samples: {len(setup_data['test_docs'])}")
    logger.info(f"  • Classes: {config['num_classes']}")
    logger.info(f"  • Graph edges: {len(setup_data['edges'])}")
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  • Epochs: {config['epochs']}")
    logger.info(f"  • Batch size: {config['batch_size']}")
    logger.info(f"  • Learning rate: {config['learning_rate']}")
    logger.info(f"  • Embedding dim: {config['embedding_dim']}")
    
    logger.info("\nNext Steps:")
    logger.info("  1. Update .env with your API keys")
    logger.info("  2. Run: python test_setup.py")
    logger.info("  3. Run: python start_finetuning.py")
    
    logger.info("\nDocumentation:")
    logger.info("  • FINETUNING_README.md - Full guide")
    logger.info("  • MULTI_AGENT_ARCHITECTURE.md - Architecture details")
    logger.info("  • data_loaders.py - Dataset utilities")
    
    logger.info("\n" + "="*70)


def main():
    """Main setup function"""
    logger.info("\n" + "█"*70)
    logger.info("█ GRAPHRAG FINE-TUNING ENVIRONMENT SETUP")
    logger.info("█"*70)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        logger.error("Failed to install dependencies. Please install manually.")
        return False
    
    # Step 2: Setup environment variables
    setup_environment_variables()
    
    # Step 3: Create project structure
    create_project_structure()
    
    # Step 4: Load dataset
    setup_data = load_firerisk_dataset()
    
    # Step 5: Create startup scripts
    create_startup_script(setup_data)
    
    # Step 6: Create documentation
    create_readme()
    
    # Step 7: Print summary
    print_summary(setup_data)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
