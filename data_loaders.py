"""
Data loading utilities for fine-tuning datasets
Supports multiple dataset sources including FireRisk and Neo4j graph integration
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

# Try to import neo4j driver for graph database support
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j package not available. Install with: pip install neo4j")


# ============================================================
# FIRERISK DATASET LOADER
# ============================================================
class FireRiskLoader:
    """Loader for FireRisk dataset from Hugging Face"""
    
    FIRE_RISK_CLASSES = {
        'high': 0,
        'low': 1,
        'moderate': 2,
        'non-burnable': 3,
        'very_high': 4,
        'very_low': 5,
        'water': 6,
    }
    
    REVERSE_CLASSES = {v: k for k, v in FIRE_RISK_CLASSES.items()}
    
    def __init__(self, cache_dir: str = "data/firerisk"):
        """Initialize FireRisk loader"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        
    def download(self, split: str = "train", limit: Optional[int] = None) -> List[Dict]:
        """
        Download FireRisk dataset from Hugging Face
        
        Args:
            split: "train" or "test"
            limit: Maximum number of samples to load (for quick prototyping)
            
        Returns:
            List of sample dictionaries with image and label
        """
        logger.info(f"Loading FireRisk dataset ({split} split)...")
        
        try:
            # Load from Hugging Face
            dataset = load_dataset("blanchon/FireRisk", split=split)
            logger.info(f"✓ Loaded {len(dataset)} samples")
            
            # Convert to list and limit if needed
            samples = []
            for i, sample in enumerate(dataset):
                if limit and i >= limit:
                    break
                    
                # Extract image and label
                image = sample.get('image')
                label_str = sample.get('split')  # The label column
                
                if label_str in self.FIRE_RISK_CLASSES:
                    label = self.FIRE_RISK_CLASSES[label_str]
                    samples.append({
                        'id': f"firerisk_{split}_{i}",
                        'content': f"Fire risk level: {label_str}. Remote sensing image analysis.",
                        'label': label,
                        'label_name': label_str,
                        'metadata': {
                            'source': 'FireRisk',
                            'split': split,
                            'image_size': (320, 320),
                            'bands': 3,
                        }
                    })
            
            logger.info(f"✓ Processed {len(samples)} samples")
            self.dataset = samples
            return samples
            
        except Exception as e:
            logger.error(f"Error loading FireRisk: {e}")
            logger.info("Falling back to mock data for testing...")
            return self._create_mock_data(limit or 100)
    
    def _create_mock_data(self, num_samples: int = 100) -> List[Dict]:
        """Create mock FireRisk data for testing when download fails"""
        logger.warning(f"Creating {num_samples} mock FireRisk samples for testing")
        
        mock_data = []
        labels = list(self.FIRE_RISK_CLASSES.keys())
        
        for i in range(num_samples):
            label_name = labels[i % len(labels)]
            label = self.FIRE_RISK_CLASSES[label_name]
            
            mock_data.append({
                'id': f"mock_firerisk_{i}",
                'content': f"Fire risk level: {label_name}. Aerial imagery from region {i % 10}.",
                'label': label,
                'label_name': label_name,
                'metadata': {
                    'source': 'FireRisk',
                    'mock': True,
                    'region': i % 10,
                }
            })
        
        return mock_data
    
    def get_label_distribution(self, samples: List[Dict]) -> Dict:
        """Get distribution of labels in dataset"""
        distribution = {}
        for sample in samples:
            label_name = sample.get('label_name', 'unknown')
            distribution[label_name] = distribution.get(label_name, 0) + 1
        return distribution


# ============================================================
# GENERIC DATASET LOADER
# ============================================================
class HuggingFaceDatasetLoader:
    """Generic loader for Hugging Face text datasets"""
    
    def __init__(self, dataset_name: str, cache_dir: str = "data"):
        """Initialize HuggingFace dataset loader"""
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_classification(
        self,
        text_column: str = "text",
        label_column: str = "label",
        split: str = "train",
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Load text classification dataset from HuggingFace
        
        Args:
            text_column: Column containing text data
            label_column: Column containing labels
            split: Dataset split to load
            limit: Max samples to load
            
        Returns:
            List of sample dicts with 'content', 'label', 'metadata'
        """
        logger.info(f"Loading {self.dataset_name} ({split})...")
        
        try:
            dataset = load_dataset(self.dataset_name, split=split)
            logger.info(f"✓ Loaded {len(dataset)} samples")
            
            samples = []
            for i, sample in enumerate(dataset):
                if limit and i >= limit:
                    break
                
                text = sample.get(text_column, "")
                label = sample.get(label_column)
                
                if text and label is not None:
                    samples.append({
                        'id': f"{self.dataset_name}_{i}",
                        'content': text,
                        'label': int(label) if isinstance(label, (int, float)) else label,
                        'metadata': {
                            'source': self.dataset_name,
                            'split': split,
                        }
                    })
            
            logger.info(f"✓ Processed {len(samples)} samples")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []


# ============================================================
# DATA PREPARATION UTILITIES
# ============================================================
def convert_to_document_format(samples: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Convert raw samples to document format expected by fine-tuning pipeline
    
    Args:
        samples: Raw samples with 'content' and 'label'
        
    Returns:
        (documents, labels) tuples
    """
    documents = []
    labels = []
    
    for sample in samples:
        doc = {
            'id': sample.get('id', f"doc_{len(documents)}"),
            'content': sample.get('content', ''),
            'metadata': sample.get('metadata', {}),
        }
        documents.append(doc)
        labels.append(sample.get('label', 0))
    
    return documents, labels


def create_graph_edges_from_documents(
    documents: List[Dict],
    edge_strategy: str = "sequential"
) -> List[Tuple[str, str]]:
    """
    Create graph edges for document retriever
    
    Args:
        documents: List of documents
        edge_strategy: 'sequential', 'content_similarity', or 'random'
        
    Returns:
        List of (source_id, target_id) tuples
    """
    edges = []
    
    if edge_strategy == "sequential":
        # Connect each document to the next
        for i in range(len(documents) - 1):
            edges.append((documents[i]['id'], documents[i + 1]['id']))
            
    elif edge_strategy == "random":
        # Random connections
        import random
        for i in range(len(documents)):
            j = random.randint(0, len(documents) - 1)
            if i != j:
                edges.append((documents[i]['id'], documents[j]['id']))
    
    return edges


# ============================================================
# SETUP UTILITIES
# ============================================================
def setup_environment():
    """Setup environment for fine-tuning"""
    import subprocess
    import sys
    
    required_packages = [
        'datasets',
        'huggingface-hub',
        'torch',
        'langchain',
        'langchain-google-genai',
        'chromadb',
        'scikit-learn',
        'matplotlib',
        'tqdm',
        'python-dotenv',
    ]
    
    logger.info("Checking and installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✓ {package} installed")


def create_firetask_setup() -> Dict:
    """Create complete FireRisk dataset setup for fine-tuning"""
    logger.info("="*70)
    logger.info("SETTING UP FIRERISK DATASET FOR FINE-TUNING")
    logger.info("="*70)
    
    # Setup environment
    setup_environment()
    
    # Download FireRisk dataset
    logger.info("\n[1] Downloading FireRisk dataset...")
    loader = FireRiskLoader(cache_dir="data/firerisk")
    
    # Load with limit for quick start
    train_samples = loader.download(split="train", limit=5000)
    test_samples = loader.download(split="test", limit=1000)
    
    logger.info(f"\n[2] Dataset Statistics:")
    logger.info(f"  • Train samples: {len(train_samples)}")
    logger.info(f"  • Test samples: {len(test_samples)}")
    
    train_dist = loader.get_label_distribution(train_samples)
    logger.info(f"  • Label distribution (train): {train_dist}")
    
    # Convert to document format
    logger.info("\n[3] Converting to document format...")
    train_docs, train_labels = convert_to_document_format(train_samples)
    test_docs, test_labels = convert_to_document_format(test_samples)
    
    # Create graph edges
    logger.info("\n[4] Creating graph edges...")
    edges = create_graph_edges_from_documents(train_docs, edge_strategy="sequential")
    logger.info(f"  • Created {len(edges)} edges")
    
    # Create config
    config = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'embedding_dim': 768,
        'embedding_model': 'models/embedding-001',
        'output_dir': 'firerisk_outputs',
        'dataset_name': 'FireRisk',
        'num_classes': 7,
    }
    
    logger.info("\n[5] Configuration:")
    for key, value in config.items():
        logger.info(f"  • {key}: {value}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ SETUP COMPLETE")
    logger.info("="*70)
    
    return {
        'train_docs': train_docs,
        'train_labels': train_labels,
        'test_docs': test_docs,
        'test_labels': test_labels,
        'edges': edges,
        'config': config,
        'label_names': loader.REVERSE_CLASSES,
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create setup
    setup = create_firetask_setup()
    
    # Print summary
    print("\nDataset ready for fine-tuning!")
    print(f"Train docs: {len(setup['train_docs'])}")
    print(f"Test docs: {len(setup['test_docs'])}")
    print(f"Graph edges: {len(setup['edges'])}")


# ============================================================
# NEO4J GRAPH DATABASE INTEGRATION
# ============================================================
class Neo4jGraphDatabase:
    """Integration with Neo4j graph database for storing and querying documents"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (e.g., "neo4j+s://db.databases.neo4j.io:7687")
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package required. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
        
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def create_document_nodes(self, documents: List[Dict]) -> int:
        """
        Create document nodes in Neo4j
        
        Args:
            documents: List of document dicts with 'id', 'content', 'label'
            
        Returns:
            Number of nodes created
        """
        count = 0
        with self.driver.session(database=self.database) as session:
            for doc in documents:
                try:
                    session.run(
                        """
                        CREATE (d:Document {
                            id: $id,
                            content: $content,
                            label: $label,
                            created_at: datetime()
                        })
                        """,
                        id=str(doc.get('id')),
                        content=doc.get('content', ''),
                        label=int(doc.get('label', 0))
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Error creating document node: {e}")
        
        logger.info(f"Created {count} document nodes in Neo4j")
        return count
    
    def create_relationships(self, edges: List[Tuple[str, str, str]]) -> int:
        """
        Create relationships between documents
        
        Args:
            edges: List of (source_id, target_id, relation_type) tuples
            
        Returns:
            Number of relationships created
        """
        count = 0
        with self.driver.session(database=self.database) as session:
            for source, target, rel_type in edges:
                try:
                    session.run(
                        f"""
                        MATCH (a:Document {{id: $source}})
                        MATCH (b:Document {{id: $target}})
                        CREATE (a)-[r:{rel_type}]->(b)
                        SET r.created_at = datetime()
                        """,
                        source=str(source),
                        target=str(target)
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Error creating relationship: {e}")
        
        logger.info(f"Created {count} relationships in Neo4j")
        return count
    
    def query_similar_documents(self, doc_id: str, relation_type: str = None, limit: int = 5) -> List[Dict]:
        """
        Query documents related to a given document
        
        Args:
            doc_id: Document ID
            relation_type: Optional relation type filter
            limit: Max results
            
        Returns:
            List of related documents
        """
        with self.driver.session(database=self.database) as session:
            if relation_type:
                query = f"""
                MATCH (d:Document {{id: $doc_id}})-[:{relation_type}]-(related:Document)
                RETURN related.id as id, related.content as content, related.label as label
                LIMIT $limit
                """
            else:
                query = """
                MATCH (d:Document {id: $doc_id})-[rel]-(related:Document)
                RETURN related.id as id, related.content as content, related.label as label, type(rel) as relation_type
                LIMIT $limit
                """
            
            result = session.run(query, doc_id=str(doc_id), limit=limit)
            return [dict(record) for record in result]
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph"""
        with self.driver.session(database=self.database) as session:
            # Count nodes
            node_count = session.run(
                "MATCH (d:Document) RETURN count(d) as count"
            ).single()['count']
            
            # Count relationships
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()['count']
            
            # Count by label
            label_stats = session.run(
                "MATCH (d:Document) RETURN d.label as label, count(*) as count"
            )
            label_counts = {record['label']: record['count'] for record in label_stats}
            
            return {
                'total_nodes': node_count,
                'total_relationships': rel_count,
                'label_distribution': label_counts
            }
    
    def clear_all(self) -> bool:
        """
        Clear all nodes and relationships from the database
        WARNING: This is destructive!
        """
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared all data from Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False


def setup_neo4j_from_env() -> Optional[Neo4jGraphDatabase]:
    """
    Create Neo4j database connection from environment variables
    
    Expects:
        NEO4J_URI: Connection URI
        NEO4J_USERNAME: Username
        NEO4J_PASSWORD: Password
        NEO4J_DATABASE: Database name (optional, defaults to 'neo4j')
    
    Returns:
        Neo4jGraphDatabase instance or None if credentials missing
    """
    uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    if not all([uri, username, password]):
        logger.info("Neo4j credentials not found in environment")
        return None
    
    try:
        return Neo4jGraphDatabase(uri, username, password, database)
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        return None


def create_documents_with_neo4j(
    documents: List[Dict],
    edges: List[Tuple[str, str, str]],
    neo4j_db: Neo4jGraphDatabase
) -> Dict:
    """
    Create documents and relationships in Neo4j
    
    Args:
        documents: List of document dicts
        edges: List of (source, target, relation_type) tuples
        neo4j_db: Neo4jGraphDatabase instance
        
    Returns:
        Summary dict with creation counts
    """
    doc_count = neo4j_db.create_document_nodes(documents)
    edge_count = neo4j_db.create_relationships(edges)
    stats = neo4j_db.get_graph_stats()
    
    return {
        'documents_created': doc_count,
        'relationships_created': edge_count,
        'graph_stats': stats
    }
