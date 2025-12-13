# Placeholder for faiss_manager.py
"""
FAISS manager for medical knowledge base
"""
import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FAISSManager:
    """Manage medical knowledge in FAISS"""
    
    def __init__(
        self,
        dimension: int = 384,
        persist_directory: str = "./data/vector_db/faiss"
    ):
        self.dimension = dimension
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Storage for documents and metadata
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        logger.info(f"FAISS index initialized with dimension {dimension}")
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """
        Add documents with embeddings to index
        
        Args:
            embeddings: numpy array of shape (n_docs, dimension)
            documents: List of text chunks
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        logger.info(f"Added {len(documents)} documents to FAISS. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filter (applied post-search)
        
        Returns:
            Dict with documents, metadatas, distances, ids
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension doesn't match index dimension")
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k * 2, self.index.ntotal))
        
        # Collect results
        results = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Apply metadata filter if specified
            if filter_dict:
                metadata = self.metadatas[idx]
                if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                    continue
            
            results['documents'].append(self.documents[idx])
            results['metadatas'].append(self.metadatas[idx])
            results['distances'].append(float(distance))
            results['ids'].append(self.ids[idx])
            
            if len(results['documents']) >= k:
                break
        
        return results
    
    def save(self):
        """Save index and metadata to disk"""
        # Save FAISS index
        index_path = self.persist_directory / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents and metadata
        data_path = self.persist_directory / "documents.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids
            }, f)
        
        logger.info(f"FAISS index saved to {self.persist_directory}")
    
    def load(self):
        """Load index and metadata from disk"""
        # Load FAISS index
        index_path = self.persist_directory / "faiss.index"
        if not index_path.exists():
            logger.warning(f"No index found at {index_path}")
            return False
        
        self.index = faiss.read_index(str(index_path))
        
        # Load documents and metadata
        data_path = self.persist_directory / "documents.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.ids = data['ids']
        
        logger.info(f"Loaded FAISS index with {self.index.ntotal} documents")
        return True
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_documents": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatL2"
        }


if __name__ == "__main__":
    # Test FAISS
    from src.vector_db.embeddings import MedicalEmbeddingGenerator
    
    faiss_mgr = FAISSManager(dimension=384)
    embed_gen = MedicalEmbeddingGenerator()
    
    # Test documents
    test_docs = [
        "Type 2 diabetes is a chronic condition affecting blood sugar regulation.",
        "Symptoms include increased thirst, frequent urination, and fatigue.",
        "Treatment involves lifestyle changes and may include medication."
    ]
    
    # Generate embeddings
    embeddings = embed_gen.generate_embeddings(test_docs)
    
    # Add to FAISS
    test_metadata = [
        {"section": "overview"},
        {"section": "symptoms"},
        {"section": "treatment"}
    ]
    test_ids = ["doc_1", "doc_2", "doc_3"]
    
    faiss_mgr.add_documents(embeddings, test_docs, test_metadata, test_ids)
    
    # Test search
    query = "What are the signs of diabetes?"
    query_emb = embed_gen.generate_single_embedding(query)
    results = faiss_mgr.search(query_emb, k=2)
    
    print("Search results:", results['documents'])
    print("Stats:", faiss_mgr.get_stats())
    
    # Test save/load
    faiss_mgr.save()