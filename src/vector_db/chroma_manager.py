# Placeholder for chroma_manager.py
"""
ChromaDB manager for medical knowledge base
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manage medical knowledge in ChromaDB"""
    
    def __init__(
        self, 
        persist_directory: str = "./data/vector_db/chroma",
        collection_name: str = "medical_knowledge"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = None
        logger.info(f"ChromaDB initialized at {self.persist_directory}")
    
    def create_collection(self, reset: bool = False):
        """Create or get collection"""
        if reset and self.collection_name in [c.name for c in self.client.list_collections()]:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Collection '{self.collection_name}' ready")
        return self.collection
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """
        Add documents to collection
        
        Args:
            documents: List of text chunks
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        if not self.collection:
            self.create_collection()
        
        try:
            # ChromaDB handles batching automatically
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Query the collection
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            filter_dict: Optional metadata filter
        
        Returns:
            Dict with documents, metadatas, distances
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_dict
            )
            
            # Reformat results
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0],
                'ids': results['ids'][0]
            }
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dimension": 384  # for all-MiniLM-L6-v2
        }


if __name__ == "__main__":
    # Test ChromaDB
    chroma = ChromaDBManager()
    chroma.create_collection(reset=True)
    
    # Add test documents
    test_docs = [
        "Type 2 diabetes is a chronic condition affecting blood sugar regulation.",
        "Symptoms include increased thirst, frequent urination, and fatigue.",
        "Treatment involves lifestyle changes and may include medication."
    ]
    
    test_metadata = [
        {"section": "overview", "source": "Medical KB"},
        {"section": "symptoms", "source": "Medical KB"},
        {"section": "treatment", "source": "Medical KB"}
    ]
    
    test_ids = ["doc_1", "doc_2", "doc_3"]
    
    chroma.add_documents(test_docs, test_metadata, test_ids)
    
    # Test query
    results = chroma.query("What are diabetes symptoms?", n_results=2)
    print("Query results:", results['documents'])
    print("Stats:", chroma.get_collection_stats())