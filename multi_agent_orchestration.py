"""
Multi-Agent Orchestration for GraphRAG Fine-Tuning
Inspired by mootboard's supervisor agent pattern
Each agent handles a specific stage of the fine-tuning pipeline
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_graph_retriever import GraphRetriever, EagerStrategy
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# AGENT BASE CLASS
# ============================================================
class Agent(ABC):
    """Base class for all agents in the pipeline"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.results = {}
        self.status = "IDLE"
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the agent's task"""
        pass
    
    def log(self, message: str):
        """Log agent messages"""
        logger.info(f"[{self.name}] {message}")
    
    def save_results(self, output_dir: Path):
        """Save agent results to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{self.name}_results.json"
        # Convert non-serializable objects
        serializable_results = {
            k: str(v) if not isinstance(v, (str, int, float, bool, dict, list)) else v
            for k, v in self.results.items()
        }
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        self.log(f"Results saved to {results_file}")


# ============================================================
# INDIVIDUAL AGENTS
# ============================================================
class DataPreprationAgent(Agent):
    """Agent 1: Prepare and validate data"""
    
    def execute(self, documents: List[Dict], labels: List[int]) -> Dict[str, Any]:
        """Prepare training, validation, and test data"""
        self.status = "PROCESSING"
        self.log("Starting data preparation...")
        
        # Validate data
        assert len(documents) == len(labels), "Documents and labels mismatch"
        num_classes = len(set(labels))
        
        self.log(f"Total documents: {len(documents)}")
        self.log(f"Number of classes: {num_classes}")
        
        # Calculate class distribution
        class_counts = Counter(labels)
        self.log(f"Class distribution: {dict(class_counts)}")
        
        # Split into train/val/test
        from sklearn.model_selection import train_test_split
        
        # 70% train, 15% val, 15% test
        train_docs, temp_docs, train_labels, temp_labels = train_test_split(
            documents, labels, test_size=0.3, stratify=labels, random_state=42
        )
        val_docs, test_docs, val_labels, test_labels = train_test_split(
            temp_docs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        self.log(f"Train: {len(train_docs)}, Val: {len(val_docs)}, Test: {len(test_docs)}")
        
        self.results = {
            'train_docs': train_docs,
            'train_labels': train_labels,
            'val_docs': val_docs,
            'val_labels': val_labels,
            'test_docs': test_docs,
            'test_labels': test_labels,
            'num_classes': num_classes,
            'class_distribution': dict(class_counts),
        }
        
        self.status = "COMPLETED"
        return self.results


class RetrieverConfigAgent(Agent):
    """Agent 2: Configure and initialize retriever"""
    
    def execute(self, edges: List[Tuple[str, str]], config: Dict) -> Dict[str, Any]:
        """Setup graph retriever with specified edges"""
        self.status = "PROCESSING"
        self.log("Configuring retriever...")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.get("embedding_model", "models/embedding-001")
        )
        self.log(f"Embeddings initialized: {config.get('embedding_model')}")
        
        # Initialize vectorstore
        vectorstore = Chroma(
            collection_name=config.get("collection_name", "finetuned_docs"),
            embedding_function=embeddings,
            chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=config.get("database_name", "graphRAG_finetuned"),
        )
        self.log("Vectorstore initialized")
        
        # Setup retriever
        retriever = GraphRetriever(
            store=vectorstore,
            edges=edges,
            strategy=EagerStrategy(),
        )
        self.log(f"Retriever configured with {len(edges)} graph edges")
        
        self.results = {
            'embeddings': embeddings,
            'vectorstore': vectorstore,
            'retriever': retriever,
            'num_edges': len(edges),
        }
        
        self.status = "COMPLETED"
        return self.results


class TrainingAgent(Agent):
    """Agent 3: Execute training loop"""
    
    def execute(
        self,
        train_loader,
        val_loader,
        num_classes: int,
        class_weights: torch.Tensor,
        config: Dict
    ) -> Dict[str, Any]:
        """Train the embedding model"""
        self.status = "TRAINING"
        self.log("Starting training...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"Device: {device}")
        
        # Setup model, loss, optimizer
        embedding_dim = config.get("embedding_dim", 768)
        epochs = config.get("epochs", 30)
        lr = config.get("learning_rate", 3e-4)
        
        clf_head = torch.nn.Linear(embedding_dim, num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(clf_head.parameters(), lr=lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        self.log(f"Model: Linear({embedding_dim} → {num_classes})")
        self.log(f"Epochs: {epochs}, LR: {lr}")
        
        history = []
        best_f1 = 0.0
        
        for epoch in range(epochs):
            # Train
            clf_head.train()
            total_loss = 0.0
            correct = 0
            total_samples = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
            for batch in pbar:
                texts = batch['text']
                labels = torch.tensor(batch['label']).to(device)
                
                # Dummy embeddings (replace with actual embeddings)
                embeddings = torch.randn(len(texts), embedding_dim).to(device)
                
                optimizer.zero_grad()
                logits = clf_head(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf_head.parameters(), 1.0)
                optimizer.step()
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size
            
            train_loss = total_loss / total_samples
            train_acc = correct / total_samples
            
            # Validate
            clf_head.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    texts = batch['text']
                    labels = torch.tensor(batch['label']).to(device)
                    
                    embeddings = torch.randn(len(texts), embedding_dim).to(device)
                    logits = clf_head(embeddings)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
            
            scheduler.step()
            
            # Log
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'lr': optimizer.param_groups[0]['lr'],
            }
            history.append(history_entry)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(clf_head.state_dict(), "best_model.pth")
                self.log(f"✓ NEW BEST F1: {val_f1:.4f}")
            
            if (epoch + 1) % 5 == 0:
                self.log(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | "
                        f"val_acc {val_acc:.3%} | val_f1 {val_f1:.4f}")
        
        self.results = {
            'model_state': clf_head.state_dict(),
            'history': history,
            'best_f1': best_f1,
        }
        
        self.status = "COMPLETED"
        return self.results


class EvaluationAgent(Agent):
    """Agent 4: Evaluate model performance"""
    
    def execute(self, model_state: Dict, test_loader, num_classes: int, config: Dict) -> Dict[str, Any]:
        """Evaluate on test set"""
        self.status = "EVALUATING"
        self.log("Starting evaluation...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = config.get("embedding_dim", 768)
        
        # Load model
        clf_head = torch.nn.Linear(embedding_dim, num_classes).to(device)
        clf_head.load_state_dict(model_state)
        clf_head.eval()
        
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                texts = batch['text']
                labels = torch.tensor(batch['label']).to(device)
                
                embeddings = torch.randn(len(texts), embedding_dim).to(device)
                logits = clf_head(embeddings)
                
                test_preds.extend(logits.argmax(dim=1).cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
        test_acc = sum([p == t for p, t in zip(test_preds, test_targets)]) / len(test_targets)
        test_precision = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
        test_recall = recall_score(test_targets, test_preds, average='weighted', zero_division=0)
        
        self.log(f"Test F1: {test_f1:.4f}")
        self.log(f"Test Accuracy: {test_acc:.3%}")
        self.log(f"Test Precision: {test_precision:.4f}")
        self.log(f"Test Recall: {test_recall:.4f}")
        
        self.results = {
            'test_f1': test_f1,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'predictions': test_preds,
            'targets': test_targets,
        }
        
        self.status = "COMPLETED"
        return self.results


class ReportingAgent(Agent):
    """Agent 5: Generate reports and visualizations"""
    
    def execute(self, training_history: List[Dict], eval_metrics: Dict) -> Dict[str, Any]:
        """Generate training reports and plots"""
        self.status = "REPORTING"
        self.log("Generating reports and visualizations...")
        
        # Plot learning curves
        epochs = [h['epoch'] for h in training_history]
        train_losses = [h['train_loss'] for h in training_history]
        val_losses = [h['val_loss'] for h in training_history]
        train_accs = [h['train_acc'] for h in training_history]
        val_accs = [h['val_acc'] for h in training_history]
        val_f1s = [h['val_f1'] for h in training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Fine-Tuning Report", fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(epochs, train_losses, 'o-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 's-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title("Training & Validation Loss")
        
        # Accuracy
        axes[0, 1].plot(epochs, train_accs, 'o-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, val_accs, 's-', label='Val', linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title("Training & Validation Accuracy")
        
        # F1 Score
        axes[1, 0].plot(epochs, val_f1s, 's-', color='green', linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Weighted F1")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title("Validation F1 Score")
        
        # Test Metrics
        metrics_text = f"""
        Test Metrics:
        • F1 Score: {eval_metrics['test_f1']:.4f}
        • Accuracy: {eval_metrics['test_acc']:.3%}
        • Precision: {eval_metrics['test_precision']:.4f}
        • Recall: {eval_metrics['test_recall']:.4f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("training_report.png", dpi=150, bbox_inches='tight')
        self.log("Report saved: training_report.png")
        plt.close()
        
        self.results = {
            'report_file': 'training_report.png',
            'summary': metrics_text,
        }
        
        self.status = "COMPLETED"
        return self.results


# ============================================================
# SUPERVISOR AGENT (ORCHESTRATOR)
# ============================================================
class SupervisorAgent:
    """
    Main orchestrator that coordinates all agents.
    Similar to mootboard's supervisor_agent pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.workflow_status = {}
        self.output_dir = Path(config.get('output_dir', 'agent_outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
    def register_agent(self, agent: Agent):
        """Register an agent in the pipeline"""
        self.agents[agent.name] = agent
        logger.info(f"✓ Agent registered: {agent.name}")
    
    def log(self, message: str):
        """Supervisor logging"""
        logger.info(f"[SUPERVISOR] {message}")
    
    def orchestrate(
        self,
        documents: List[Dict],
        labels: List[int],
        edges: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline:
        1. Data Preparation
        2. Retriever Configuration
        3. Training
        4. Evaluation
        5. Reporting
        """
        
        self.log("="*70)
        self.log("STARTING MULTI-AGENT FINE-TUNING PIPELINE")
        self.log("="*70)
        
        # Step 1: Data Preparation
        self.log("\n[1/5] Data Preparation Agent")
        prep_agent = self.agents.get('data_prep')
        data_results = prep_agent.execute(documents, labels)
        prep_agent.save_results(self.output_dir)
        self.workflow_status['data_prep'] = 'COMPLETED'
        
        # Step 2: Retriever Configuration
        self.log("\n[2/5] Retriever Configuration Agent")
        retriever_agent = self.agents.get('retriever_config')
        retriever_results = retriever_agent.execute(edges, self.config)
        retriever_agent.save_results(self.output_dir)
        self.workflow_status['retriever_config'] = 'COMPLETED'
        
        # Step 3: Create data loaders
        self.log("\n[Preparing DataLoaders]")
        from fine_tune import create_data_loaders, DocumentDataset
        
        train_loader, val_loader, test_loader, num_classes = create_data_loaders(
            data_results['train_docs'], data_results['train_labels'],
            data_results['val_docs'], data_results['val_labels'],
            data_results['test_docs'], data_results['test_labels'],
            batch_size=self.config.get('batch_size', 32)
        )
        
        train_dataset = DocumentDataset(data_results['train_docs'], data_results['train_labels'])
        class_weights = train_dataset.get_class_weights()
        
        # Step 4: Training
        self.log("\n[3/5] Training Agent")
        training_agent = self.agents.get('training')
        training_results = training_agent.execute(
            train_loader, val_loader, num_classes, class_weights, self.config
        )
        training_agent.save_results(self.output_dir)
        self.workflow_status['training'] = 'COMPLETED'
        
        # Step 5: Evaluation
        self.log("\n[4/5] Evaluation Agent")
        eval_agent = self.agents.get('evaluation')
        eval_results = eval_agent.execute(
            training_results['model_state'], test_loader, num_classes, self.config
        )
        eval_agent.save_results(self.output_dir)
        self.workflow_status['evaluation'] = 'COMPLETED'
        
        # Step 6: Reporting
        self.log("\n[5/5] Reporting Agent")
        reporting_agent = self.agents.get('reporting')
        report_results = reporting_agent.execute(
            training_results['history'], eval_results
        )
        reporting_agent.save_results(self.output_dir)
        self.workflow_status['reporting'] = 'COMPLETED'
        
        # Final Summary
        self._print_summary(eval_results, data_results)
        
        return {
            'data': data_results,
            'retriever': retriever_results,
            'training': training_results,
            'evaluation': eval_results,
            'report': report_results,
            'workflow_status': self.workflow_status,
        }
    
    def _print_summary(self, eval_results: Dict, data_results: Dict):
        """Print final summary"""
        self.log("\n" + "="*70)
        self.log("PIPELINE COMPLETED SUCCESSFULLY ✓")
        self.log("="*70)
        self.log(f"\nDataset Summary:")
        self.log(f"  • Classes: {data_results['num_classes']}")
        self.log(f"  • Distribution: {data_results['class_distribution']}")
        self.log(f"\nFinal Metrics:")
        self.log(f"  • Test F1: {eval_results['test_f1']:.4f}")
        self.log(f"  • Test Accuracy: {eval_results['test_acc']:.3%}")
        self.log(f"  • Test Precision: {eval_results['test_precision']:.4f}")
        self.log(f"  • Test Recall: {eval_results['test_recall']:.4f}")
        self.log(f"\nArtifacts:")
        self.log(f"  • Outputs: {self.output_dir}")
        self.log(f"  • Report: training_report.png")
        self.log(f"  • Model: best_model.pth")
        self.log("="*70)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def setup_pipeline(
    documents: List[Dict],
    labels: List[int],
    edges: List[Tuple[str, str]],
    config: Dict[str, Any],
) -> SupervisorAgent:
    """
    Setup the multi-agent pipeline
    """
    
    # Create supervisor
    supervisor = SupervisorAgent(config)
    
    # Register agents in order
    supervisor.register_agent(
        DataPreprationAgent("data_prep", config)
    )
    supervisor.register_agent(
        RetrieverConfigAgent("retriever_config", config)
    )
    supervisor.register_agent(
        TrainingAgent("training", config)
    )
    supervisor.register_agent(
        EvaluationAgent("evaluation", config)
    )
    supervisor.register_agent(
        ReportingAgent("reporting", config)
    )
    
    return supervisor


if __name__ == "__main__":
    print("Multi-Agent Orchestration module loaded")
    print("Use setup_pipeline() to create and execute the pipeline")
