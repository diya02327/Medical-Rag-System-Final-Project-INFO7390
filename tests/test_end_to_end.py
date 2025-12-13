"""
End-to-end integration tests
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import os
from dotenv import load_dotenv
from src.data_collection.dataset_loader import MedicalDatasetLoader
from src.preprocessing.document_processor import MedicalDocumentProcessor
from src.vector_db.chroma_manager import ChromaDBManager
from src.llm.rag_pipeline import MedicalRAGPipeline
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_complete_pipeline(self):
        """Test complete pipeline from data to response"""
        logger.info("Testing complete pipeline...")
        
        # Step 1: Load data
        loader = MedicalDatasetLoader()
        documents = loader.create_sample_medical_dataset()
        self.assertGreater(len(documents), 0, "Should load documents")
        
        # Step 2: Process documents
        processor = MedicalDocumentProcessor()
        processed_docs = processor.process_batch(documents)
        self.assertGreater(len(processed_docs), 0, "Should process documents")
        
        # Step 3: Build vector database
        chroma = ChromaDBManager()
        chroma.create_collection(reset=True)
        
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc['chunks'])
        
        documents_text = [chunk['text'] for chunk in all_chunks]
        metadatas = [chunk['metadata'] for chunk in all_chunks]
        ids = [chunk['chunk_id'] for chunk in all_chunks]
        
        chroma.add_documents(documents_text, metadatas, ids)
        
        # Step 4: Query RAG pipeline
        rag = MedicalRAGPipeline(
            vector_db_manager=chroma,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        result = rag.query("What are the symptoms of diabetes?", n_results=3)
        
        # Verify complete result
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertGreater(len(result['answer']), 50)
        self.assertGreater(len(result['sources']), 0)
        
        logger.info("✅ Complete pipeline test passed")


def run_e2e_tests():
    """Run end-to-end tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEndToEnd)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)