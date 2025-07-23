import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path


class FAISSVectorStore:
    """FAISS-based vector store for MTG rules and cards with RAG functionality."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_dir: str = "data/vector_index",
                 dimension: int = 384):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.dimension = dimension
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Storage for metadata
        self.documents = []
        self.metadata = []
        
        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        if len(documents) != len(metadata):
            raise ValueError("Documents and metadata must have the same length")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Encode documents
        embeddings = self._encode_texts(documents)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def add_rules(self, rules: List[Dict[str, Any]]):
        """Add MTG rules to the vector store."""
        documents = []
        metadata = []
        
        for rule in rules:
            # Create searchable text from rule
            text = f"Rule {rule.get('id', '')}: {rule.get('content', '')}"
            documents.append(text)
            
            # Add metadata
            meta = {
                "type": "rule",
                "rule_id": rule.get("id"),
                "content": rule.get("content"),
                "source": "comprehensive_rules"
            }
            metadata.append(meta)
        
        self.add_documents(documents, metadata)
    
    def add_cards(self, cards: List[Dict[str, Any]]):
        """Add MTG cards to the vector store."""
        documents = []
        metadata = []
        
        for card in cards:
            # Create searchable text from card
            text_parts = [card.get("name", "")]
            
            if card.get("oracle_text"):
                text_parts.append(card["oracle_text"])
            
            if card.get("type_line"):
                text_parts.append(card["type_line"])
            
            text = " ".join(text_parts)
            documents.append(text)
            
            # Add metadata
            meta = {
                "type": "card",
                "name": card.get("name"),
                "oracle_id": card.get("oracle_id"),
                "oracle_text": card.get("oracle_text"),
                "type_line": card.get("type_line"),
                "mana_cost": card.get("mana_cost"),
                "source": "cards"
            }
            metadata.append(meta)
        
        self.add_documents(documents, metadata)
    
    def add_rulings(self, rulings: List[Dict[str, Any]]):
        """Add MTG rulings to the vector store."""
        documents = []
        metadata = []
        
        for ruling in rulings:
            # Create searchable text from ruling
            card_name = ruling.get("card_name", "Unknown Card")
            comment = ruling.get("comment", "")
            text = f"{card_name}: {comment}"
            documents.append(text)
            
            # Add metadata
            meta = {
                "type": "ruling",
                "card_name": card_name,
                "oracle_id": ruling.get("oracle_id"),
                "comment": comment,
                "published_at": ruling.get("published_at"),
                "source": "rulings"
            }
            metadata.append(meta)
        
        self.add_documents(documents, metadata)
    
    def search(self, query: str, k: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self._encode_texts([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, min(k * 2, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):  # Valid index
                meta = self.metadata[idx].copy()
                meta["similarity_score"] = float(score)
                meta["document"] = self.documents[idx]
                
                # Apply type filter if specified
                if filter_type is None or meta.get("type") == filter_type:
                    results.append(meta)
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        return results
    
    def search_rules(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search specifically for rules."""
        return self.search(query, k, filter_type="rule")
    
    def search_cards(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search specifically for cards."""
        return self.search(query, k, filter_type="card")
    
    def search_rulings(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search specifically for rulings."""
        return self.search(query, k, filter_type="ruling")
    
    def get_retrieval_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for RAG retrieval."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_rules",
                    "description": "Search Magic: The Gathering comprehensive rules for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant rules"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_cards",
                    "description": "Search for Magic: The Gathering cards and their text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", 
                                "description": "Search query for finding relevant cards"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_rulings",
                    "description": "Search for official rulings and clarifications about Magic cards",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant rulings"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer."""
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        return embeddings.astype('float32')
    
    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            index_path = self.index_dir / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents and metadata
            with open(self.index_dir / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(self.index_dir / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            
            # Save config
            config = {
                "model_name": self.model_name,
                "dimension": self.dimension,
                "num_documents": len(self.documents)
            }
            with open(self.index_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved vector store index to {self.index_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store index: {e}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata from disk."""
        try:
            index_path = self.index_dir / "faiss.index"
            
            if not index_path.exists():
                logger.info("No existing vector store index found")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents and metadata
            with open(self.index_dir / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            
            with open(self.index_dir / "metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded vector store with {len(self.documents)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.metadata = []
    
    def clear(self):
        """Clear the vector store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        type_counts = {}
        for meta in self.metadata:
            doc_type = meta.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "type_counts": type_counts
        }