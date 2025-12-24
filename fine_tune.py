"""
Fine-tuning module for graphRAG - Adapted from skinLesion_VIT project
Handles fine-tuning of embeddings and retriever models with comprehensive
training tracking, checkpointing, and evaluation
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
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

load_dotenv()


# ============================================================
# CONFIG & SETUP
# ============================================================
class FineTuneConfig:
    """Configuration for fine-tuning"""
    
    # Training params
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.05
    SEED = 42
    
    # Model params
    MODEL_NAME = "models/embedding-001"
    EMBEDDING_DIM = 768
    IMG_SIZE = 256  # For consistency with skin lesion project
    
    # Loss & regularization
    ALPHA = 0.5  # Weighted loss parameter
    MAX_GRAD_NORM = 1.0
    
    # Checkpoint settings
    CHECKPOINT_DIR = Path("checkpoints")
    BEST_MODEL_PATH = "best_model_f1.pth"
    
    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============================================================
# DATASET PREPARATION
# ============================================================
class DocumentDataset:
    """Dataset handler for documents with labels"""
    
    def __init__(self, documents: List[Dict], labels: List[int]):
        self.documents = documents
        self.labels = labels
        self.num_classes = len(set(labels))
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return {
            'text': self.documents[idx]['content'],
            'label': self.labels[idx],
            'metadata': self.documents[idx].get('metadata', {})
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        counter = Counter(self.labels)
        total = len(self.labels)
        weights = torch.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            count = counter.get(class_idx, 1)
            weights[class_idx] = total / (self.num_classes * count)
        
        return weights / weights.sum() * self.num_classes


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
class EmbeddingFinetuner:
    """Fine-tuner for embeddings and retriever"""
    
    def __init__(self, config: FineTuneConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.MODEL_NAME
        )
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            collection_name="finetuned_docs",
            embedding_function=self.embeddings,
            chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database="graphRAG_finetuned",
        )
        
        # Retriever
        self.retriever = None
        
        # Training state
        self.history = []
        self.best_f1 = 0.0
        self.optimizer = None
        self.scheduler = None
        
    def setup_retriever(self, edges: List[Tuple[str, str]]):
        """Setup graph traversal retriever"""
        self.retriever = GraphRetriever(
            store=self.vectorstore,
            edges=edges,
            strategy=EagerStrategy(),
        )
    
    def train_epoch(
        self,
        train_loader,
        criterion,
        epoch: int,
        log_interval: int = 50
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch:02d}",
            leave=False,
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for batch_idx, batch in pbar:
            texts = batch['text']
            labels = torch.tensor(batch['label']).to(self.device)
            
            # Get embeddings
            embeddings = self.embeddings.embed_documents(texts)
            embeddings = torch.tensor(embeddings).to(self.device)
            
            # Dummy classification head (would be replaced with actual model)
            # For demonstration, using a simple linear layer
            if not hasattr(self, 'clf_head'):
                self.clf_head = torch.nn.Linear(
                    self.config.EMBEDDING_DIM,
                    self.vectorstore._collection.metadata.get('num_classes', 2)
                ).to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            logits = self.clf_head(embeddings)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.clf_head.parameters(),
                self.config.MAX_GRAD_NORM
            )
            self.optimizer.step()
            
            # Metrics
            batch_size = labels.size(0)
            batch_loss = loss.item()
            batch_acc = (logits.argmax(dim=1) == labels).sum().item() / batch_size
            
            total_loss += batch_loss * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
            
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            if batch_idx % log_interval == 0:
                pbar.update(log_interval)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_acc = correct / total_samples if total_samples > 0 else 0.0
        avg_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, avg_acc, avg_f1
    
    def evaluate(
        self,
        eval_loader,
        criterion,
        num_classes: int
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        
        self.clf_head.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                texts = batch['text']
                labels = torch.tensor(batch['label']).to(self.device)
                
                embeddings = self.embeddings.embed_documents(texts)
                embeddings = torch.tensor(embeddings).to(self.device)
                
                logits = self.clf_head(embeddings)
                loss = criterion(logits, labels)
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size
                
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_acc = correct / total_samples if total_samples > 0 else 0.0
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # AUC calculation (for binary/multi-class)
        try:
            auc = roc_auc_score(
                all_targets,
                all_probs,
                multi_class='ovr',
                average='weighted',
                zero_division=0
            )
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.clf_head.state_dict() if hasattr(self, 'clf_head') else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'metrics': metrics,
            'history': self.history,
        }
        
        # Save regular checkpoint
        ckpt_path = self.config.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"  [Save] Checkpoint: {ckpt_path.name}")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.config.BEST_MODEL_PATH)
            print(f"  [✓] NEW BEST F1 = {metrics['weighted_f1']:.4f} -> {self.config.BEST_MODEL_PATH}")
    
    def plot_learning_curves(self):
        """Plot and save learning curves"""
        
        epochs = [h['epoch'] for h in self.history]
        tr_losses = [h['train_loss'] for h in self.history]
        val_losses = [h['val_loss'] for h in self.history]
        tr_accs = [h['train_acc'] for h in self.history]
        val_accs = [h['val_acc'] for h in self.history]
        val_f1s = [h['val_f1'] for h in self.history]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Fine-Tuning Learning Curves", fontsize=16)
        
        # Loss
        ax1.plot(epochs, tr_losses, 'o-', label='Train', linewidth=2)
        ax1.plot(epochs, val_losses, 's-', label='Val', linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, tr_accs, 'o-', label='Train', linewidth=2)
        ax2.plot(epochs, val_accs, 's-', label='Val', linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score
        ax3.plot(epochs, val_f1s, 's-', label='Val F1', linewidth=2, color='green')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Weighted F1")
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig("learning_curves.png", dpi=150, bbox_inches='tight')
        print("\nLearning curves saved: learning_curves.png")
        plt.show()
    
    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_classes: int,
        class_weights: torch.Tensor
    ):
        """Full training pipeline"""
        
        # Setup criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Classification head
        self.clf_head = torch.nn.Linear(
            self.config.EMBEDDING_DIM,
            num_classes
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.clf_head.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.EPOCHS,
            eta_min=1e-6
        )
        
        # Load existing checkpoint if available
        start_epoch = 0
        if Path(self.config.BEST_MODEL_PATH).exists():
            ckpt = torch.load(self.config.BEST_MODEL_PATH, map_location=self.device)
            self.clf_head.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            self.best_f1 = ckpt['best_f1']
            self.history = ckpt['history']
            print(f"Resumed from epoch {start_epoch}, best_f1={self.best_f1:.4f}")
        else:
            print("No checkpoint found. Starting from scratch.")
        
        print("\n" + "="*90)
        print("STARTING FINE-TUNING – Live Logs + Auto-Resume + Final Plots")
        print("="*90)
        
        # Training loop
        for epoch in range(start_epoch, self.config.EPOCHS):
            self.clf_head.train()
            
            # Train
            tr_loss, tr_acc, tr_f1 = self.train_epoch(train_loader, criterion, epoch)
            
            # Validate
            self.clf_head.eval()
            val_metrics = self.evaluate(val_loader, criterion, num_classes)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log history
            history_entry = {
                'epoch': epoch,
                'train_loss': tr_loss,
                'train_acc': tr_acc,
                'train_f1': tr_f1,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['weighted_f1'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            }
            self.history.append(history_entry)
            
            # Check if best
            is_best = val_metrics['weighted_f1'] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics['weighted_f1']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch:02d} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3%} | "
                  f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3%} | "
                  f"F1 {val_metrics['weighted_f1']:.4f} AUC {val_metrics['auc']:.4f} | "
                  f"lr {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Final evaluation on test set
        print("\n" + "="*90)
        print("LOADING BEST MODEL & FINAL TEST SET EVALUATION")
        print("="*90)
        
        best_ckpt = torch.load(self.config.BEST_MODEL_PATH, map_location=self.device)
        self.clf_head.load_state_dict(best_ckpt['model_state'])
        
        test_metrics = self.evaluate(test_loader, criterion, num_classes)
        
        print(f"\nTest Loss:        {test_metrics['loss']:.4f}")
        print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
        print(f"Test Macro F1:    {test_metrics['macro_f1']:.4f}")
        print(f"Test AUC:         {test_metrics['auc']:.4f}")
        print(f"Test Precision:   {test_metrics['precision']:.4f}")
        print(f"Test Recall:      {test_metrics['recall']:.4f}")
        
        # Plot learning curves
        self.plot_learning_curves()
        
        print("\n" + "="*90)
        print(f"Training complete! Best weighted F1: {self.best_f1:.4f}")
        print(f"Checkpoints in: {self.config.CHECKPOINT_DIR}")
        print(f"Best model: {self.config.BEST_MODEL_PATH}")
        print("="*90)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def create_data_loaders(
    train_docs: List[Dict],
    train_labels: List[int],
    val_docs: List[Dict],
    val_labels: List[int],
    test_docs: List[Dict],
    test_labels: List[int],
    batch_size: int = 32
) -> Tuple:
    """Create data loaders from documents and labels"""
    
    train_dataset = DocumentDataset(train_docs, train_labels)
    val_dataset = DocumentDataset(val_docs, val_labels)
    test_dataset = DocumentDataset(test_docs, test_labels)
    
    # Weighted sampling for imbalanced data
    class_weights = train_dataset.get_class_weights()
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes


# ============================================================
# MULTI-AGENT INTEGRATION
# ============================================================
def get_default_config() -> Dict:
    """Get default configuration for multi-agent pipeline"""
    config = FineTuneConfig()
    return {
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'embedding_dim': config.EMBEDDING_DIM,
        'embedding_model': config.MODEL_NAME,
        'collection_name': 'finetuned_docs',
        'database_name': 'graphRAG_finetuned',
        'output_dir': 'agent_outputs',
        'weight_decay': config.WEIGHT_DECAY,
        'max_grad_norm': config.MAX_GRAD_NORM,
    }


def run_multi_agent_pipeline(
    documents: List[Dict],
    labels: List[int],
    edges: List[Tuple[str, str]],
    config: Optional[Dict] = None,
) -> Dict:
    """
    Execute the multi-agent fine-tuning pipeline
    
    Usage:
        from multi_agent_orchestration import setup_pipeline
        from fine_tune import run_multi_agent_pipeline
        
        result = run_multi_agent_pipeline(
            documents=my_docs,
            labels=my_labels,
            edges=graph_edges
        )
    """
    from multi_agent_orchestration import setup_pipeline
    
    if config is None:
        config = get_default_config()
    
    supervisor = setup_pipeline(documents, labels, edges, config)
    return supervisor.orchestrate(documents, labels, edges)


if __name__ == "__main__":
    print("Fine-tuning module loaded")
    print("Use create_data_loaders(), EmbeddingFinetuner, or run_multi_agent_pipeline()")
