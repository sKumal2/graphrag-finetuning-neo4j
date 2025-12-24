#!/usr/bin/env python3
"""
GraphRAG Fine-Tuning: Quick Reference & Getting Started
Run this file for an interactive setup guide
"""

import sys
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          GraphRAG Multi-Agent Fine-Tuning Environment               ║
║                                                                      ║
║          Powered by: LangChain + PyTorch + FireRisk Dataset         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_quick_start():
    """Print quick start instructions"""
    quick_start = """
📋 QUICK START (3 Commands)
═══════════════════════════════════════════════════════════════════════

1️⃣  SETUP
    $ python finetune_setup.py
    
    This will:
    ✓ Check dependencies
    ✓ Download FireRisk dataset (5k samples)
    ✓ Create project directories
    ✓ Generate startup scripts
    
    ⏱️  Takes ~2-5 minutes (depending on internet)

2️⃣  VERIFY
    $ python test_setup.py
    
    This will:
    ✓ Test dataset loading
    ✓ Verify configuration
    ✓ Create sample batches
    ✓ Check all systems
    
    ⏱️  Takes ~30 seconds

3️⃣  TRAIN
    $ python start_finetuning.py
    
    This will:
    ✓ Execute multi-agent pipeline
    ✓ Train embedding classifier
    ✓ Evaluate on test set
    ✓ Generate visualizations
    
    ⏱️  Takes ~5-15 minutes (depends on hardware)

═══════════════════════════════════════════════════════════════════════
"""
    print(quick_start)


def print_file_guide():
    """Print file structure guide"""
    guide = """
📂 FILE STRUCTURE
═══════════════════════════════════════════════════════════════════════

Core Modules:
  fine_tune.py ..................... Main fine-tuning trainer
  multi_agent_orchestration.py ..... Agent classes & supervisor
  data_loaders.py .................. Dataset loading utilities
  
Startup Scripts:
  finetune_setup.py ................ Complete setup (RUN FIRST!)
  start_finetuning.py .............. Begin fine-tuning
  test_setup.py .................... Verify installation
  
Configuration:
  requirements.txt ................. Dependencies
  .env ............................ API keys (CONFIGURE THIS!)
  
Documentation:
  SETUP_GUIDE.md ................... This guide (READ THIS!)
  MULTI_AGENT_ARCHITECTURE.md ...... Architecture details
  FINETUNING_README.md ............. Usage examples
  example_multi_agent_finetune.py .. Code examples
  
Runtime Directories (Created automatically):
  data/firerisk/ ................... Dataset cache
  checkpoints/ ..................... Model checkpoints
  agent_outputs/ ................... Agent results (JSON)
  logs/ ........................... Training logs
  visualizations/ .................. Generated plots

═══════════════════════════════════════════════════════════════════════
"""
    print(guide)


def print_dataset_info():
    """Print dataset information"""
    info = """
📊 DATASET: FireRisk (Hugging Face)
═══════════════════════════════════════════════════════════════════════

Source: https://huggingface.co/datasets/blanchon/FireRisk

Dataset Details:
  • Total Images: 91,872 (using 5,000 for quick start)
  • Image Size: 320×320 pixels
  • Bands: 3 (RGB)
  • Classes: 7 (fire risk levels)
  • Resolution: 1m
  • Source: NAIP Aerial Imagery
  
Classes:
  0: high          (highest fire risk)
  1: low           
  2: moderate      
  3: non-burnable  
  4: very_high     (highest risk)
  5: very_low      
  6: water         (no fire risk)

Splits:
  Train: 4,000 samples (80%)
  Val:   500 samples (10%)
  Test:  500 samples (10%)

Note: Images converted to text documents for embedding fine-tuning
      (you can integrate actual image features later)

═══════════════════════════════════════════════════════════════════════
"""
    print(info)


def print_architecture():
    """Print architecture overview"""
    arch = """
🏗️  MULTI-AGENT ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

Pipeline Flow:
┌─────────────────────────────────────────────┐
│     SUPERVISOR AGENT (Orchestrator)         │
│  • Manages workflow                         │
│  • Coordinates agents                       │
│  • Aggregates results                       │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        ↓                 ↓
   ┌─────────┐      ┌──────────┐
   │DataPrep │      │Retriever │
   │ Agent   │      │ Config   │
   │         │      │ Agent    │
   └────┬────┘      └────┬─────┘
        │                │
        └────────┬───────┘
                 ↓
            ┌─────────┐
            │Training │
            │ Agent   │
            └────┬────┘
                 ↓
            ┌─────────┐
            │Evaluation
            │ Agent   │
            └────┬────┘
                 ↓
            ┌──────────┐
            │Reporting │
            │ Agent    │
            └──────────┘

Agents (Executed in Sequence):

1. DataPrepationAgent
   Input: Documents + Labels
   Output: Train/Val/Test splits
   
2. RetrieverConfigAgent
   Input: Graph edges
   Output: Embeddings + Vectorstore + Retriever
   
3. TrainingAgent
   Input: Data loaders + Config
   Output: Trained model + History
   
4. EvaluationAgent
   Input: Model + Test data
   Output: Metrics (F1, Accuracy, etc.)
   
5. ReportingAgent
   Input: History + Metrics
   Output: Visualizations + Summary

═══════════════════════════════════════════════════════════════════════
"""
    print(arch)


def print_configuration():
    """Print configuration guide"""
    config = """
⚙️  CONFIGURATION & API KEYS
═══════════════════════════════════════════════════════════════════════

Required: Google Generative AI (for embeddings)
├─ Get from: https://console.cloud.google.com/
├─ Add to .env:
│  GOOGLE_API_KEY=your_api_key_here
└─ Free tier: ✓ Supports fine-tuning

Optional: Chroma Vector Database
├─ For cloud storage (local is default)
├─ Get from: https://trychroma.com/
├─ Add to .env:
│  CHROMA_API_KEY=your_key
│  CHROMA_TENANT=default
└─ Can be skipped (uses local SQLite)

Optional: Hugging Face Token
├─ For faster dataset downloads
├─ Get from: https://huggingface.co/settings/tokens
├─ Add to .env:
│  HUGGINGFACE_TOKEN=your_token
└─ Can be skipped (public datasets work fine)

Training Hyperparameters (customize in start_finetuning.py):
├─ epochs: 30 (training epochs)
├─ batch_size: 32 (batch size)
├─ learning_rate: 3e-4 (Adam LR)
├─ weight_decay: 0.05 (L2 regularization)
├─ embedding_dim: 768 (embedding dimension)
└─ max_grad_norm: 1.0 (gradient clipping)

═══════════════════════════════════════════════════════════════════════
"""
    print(config)


def print_troubleshooting():
    """Print troubleshooting guide"""
    trouble = """
🔧 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════

❌ Error: "Missing GOOGLE_API_KEY"
   ✓ Add GOOGLE_API_KEY to .env file
   ✓ Get key from Google Cloud Console
   
❌ Error: "CUDA out of memory"
   ✓ Reduce batch_size in start_finetuning.py
   ✓ Change: config['batch_size'] = 8 (from 32)
   
❌ Error: "Dataset download failed"
   ✓ Script automatically uses mock data
   ✓ Can use local dataset instead (see FINETUNING_README.md)
   
❌ Error: "Module not found"
   ✓ Install dependencies: pip install -r requirements.txt
   ✓ Or run: python finetune_setup.py
   
❌ Slow training
   ✓ Reduce epochs: config['epochs'] = 5
   ✓ Reduce dataset: limit=1000 in loader
   ✓ Use smaller batch_size
   
⚠️  Training takes too long?
   ✓ GPU: Should be ~5-15 minutes
   ✓ CPU: Will be much slower (use GPU if possible)
   ✓ Check: python -c "import torch; print(torch.cuda.is_available())"

═══════════════════════════════════════════════════════════════════════
"""
    print(trouble)


def print_next_steps():
    """Print next steps"""
    next_steps = """
✅ NEXT STEPS
═══════════════════════════════════════════════════════════════════════

1. READ
   □ SETUP_GUIDE.md (comprehensive guide)
   □ MULTI_AGENT_ARCHITECTURE.md (architecture details)
   □ FINETUNING_README.md (usage examples)

2. CONFIGURE
   □ Create .env file
   □ Add GOOGLE_API_KEY
   □ (Optional) Add CHROMA_API_KEY

3. SETUP
   □ Run: python finetune_setup.py
   □ Wait for completion

4. VERIFY
   □ Run: python test_setup.py
   □ Verify all tests pass

5. TRAIN
   □ Run: python start_finetuning.py
   □ Monitor progress
   □ Check artifacts in agent_outputs/

6. INTEGRATE
   □ Use trained embeddings in GraphRAG
   □ Deploy model for inference
   □ Monitor with W&B or MLflow

═══════════════════════════════════════════════════════════════════════
"""
    print(next_steps)


def print_learning_resources():
    """Print learning resources"""
    resources = """
📚 LEARNING RESOURCES
═══════════════════════════════════════════════════════════════════════

Official Documentation:
  • LangChain: https://python.langchain.com
  • PyTorch: https://pytorch.org/docs
  • Chroma: https://docs.trychroma.com
  • scikit-learn: https://scikit-learn.org
  
Papers & Articles:
  • FireRisk Paper: https://arxiv.org/abs/2303.07035
  • RAG Overview: https://arxiv.org/abs/2005.11401
  • Fine-tuning Guide: https://huggingface.co/docs/transformers/training
  
Tutorials:
  • LangChain Docs: https://docs.langchain.com/docs/
  • PyTorch Tutorials: https://pytorch.org/tutorials/
  • Vector DB Guide: https://docs.trychroma.com/guide
  
Community:
  • HuggingFace Hub: https://huggingface.co
  • GitHub: https://github.com/langchain-ai/langchain
  • Discord: LangChain community Discord

═══════════════════════════════════════════════════════════════════════
"""
    print(resources)


def main():
    """Main function - print all guides"""
    print_banner()
    print_quick_start()
    print_file_guide()
    print_dataset_info()
    print_architecture()
    print_configuration()
    print_troubleshooting()
    print_next_steps()
    print_learning_resources()
    
    print("\n" + "="*73)
    print("Ready to get started? 🚀")
    print("="*73)
    print("\nStep 1: python finetune_setup.py")
    print("Step 2: python test_setup.py")
    print("Step 3: python start_finetuning.py")
    print("\n" + "="*73 + "\n")


if __name__ == "__main__":
    main()
