# Placeholder for test_retrieval.py
"""
Test retrieval quality of the vector databases
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
from typing import List, Dict
import numpy as np
from src.vector_db.chroma_manager import ChromaDBManager
from src.vector_db.faiss_manager import FAISSManager
from src.vector_db.embeddings import MedicalEmbeddingGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRetrievalQuality(unittest.TestCase):
    """Test retrieval quality metrics"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test databases"""
        cls.chroma = ChromaDBManager(
            persist_directory="./data/vector_db/chroma"
        )
        cls.chroma.create_collection()
        
        cls.embed_gen = MedicalEmbeddingGenerator()
        
        # Test queries with expected relevant terms
        cls.test_cases = [
            {
                "query": "What are the symptoms of diabetes?",
                "expected_terms": ["diabetes", "symptoms", "thirst", "urination", "fatigue"],
                "expected_source": ["diabetes", "type 2"]
            },
            {
                "query": "How can I prevent high blood pressure?",
                "expected_terms": ["blood pressure", "prevent", "diet", "exercise", "lifestyle"],
                "expected_source": ["hypertension", "blood pressure"]
            },
            {
                "query": "What causes migraine headaches?",
                "expected_terms": ["migraine", "headache", "causes", "triggers", "pain"],
                "expected_source": ["migraine"]
            },
            {
                "query": "Anxiety disorder treatment options",
                "expected_terms": ["anxiety", "treatment", "therapy", "medication"],
                "expected_source": ["anxiety"]
            }
        ]
    
    def test_retrieval_relevance(self):
        """Test if retrieved documents are relevant"""
        logger.info("Testing retrieval relevance...")
        
        for test_case in self.test_cases:
            with self.subTest(query=test_case["query"]):
                results = self.chroma.query(test_case["query"], n_results=5)
                
                # Check if we got results
                self.assertGreater(len(results['documents']), 0, 
                                 f"No results for query: {test_case['query']}")
                
                # Check if expected terms appear in results
                combined_text = " ".join(results['documents']).lower()
                
                matches = sum(1 for term in test_case['expected_terms'] 
                            if term.lower() in combined_text)
                
                match_rate = matches / len(test_case['expected_terms'])
                
                logger.info(f"Query: {test_case['query']}")
                logger.info(f"Term match rate: {match_rate:.2%}")
                
                self.assertGreater(match_rate, 0.4, 
                                 f"Low relevance for query: {test_case['query']}")
    
    def test_retrieval_diversity(self):
        """Test if retrieved documents are diverse"""
        logger.info("Testing retrieval diversity...")
        
        for test_case in self.test_cases:
            results = self.chroma.query(test_case["query"], n_results=5)
            
            # Check diversity by comparing document similarity
            documents = results['documents']
            
            if len(documents) > 1:
                unique_content = set()
                for doc in documents:
                    # Use first 100 chars as signature
                    signature = doc[:100]
                    unique_content.add(signature)
                
                diversity_ratio = len(unique_content) / len(documents)
                
                logger.info(f"Diversity ratio for '{test_case['query']}': {diversity_ratio:.2%}")
                
                self.assertGreater(diversity_ratio, 0.7,
                                 "Retrieved documents are too similar")
    
    def test_retrieval_speed(self):
        """Test retrieval speed"""
        import time
        
        logger.info("Testing retrieval speed...")
        
        query = "What are common symptoms?"
        
        # Test ChromaDB speed
        start = time.time()
        for _ in range(10):
            self.chroma.query(query, n_results=5)
        chroma_time = (time.time() - start) / 10
        
        logger.info(f"Average ChromaDB query time: {chroma_time:.3f}s")
        
        # Speed should be reasonable (< 1 second)
        self.assertLess(chroma_time, 1.0,
                       "ChromaDB queries are too slow")
    
    def test_metadata_preservation(self):
        """Test if metadata is preserved correctly"""
        logger.info("Testing metadata preservation...")
        
        results = self.chroma.query("diabetes symptoms", n_results=3)
        
        for metadata in results['metadatas']:
            # Check required fields
            self.assertIn('source', metadata)
            self.assertIn('section', metadata)
            self.assertIn('document_title', metadata)
            
            logger.info(f"Metadata: {metadata}")


class TestEmbeddingQuality(unittest.TestCase):
    """Test embedding quality"""
    
    @classmethod
    def setUpClass(cls):
        cls.embed_gen = MedicalEmbeddingGenerator()
    
    def test_embedding_similarity(self):
        """Test if similar texts have similar embeddings"""
        logger.info("Testing embedding similarity...")
        
        # Similar texts
        text1 = "Diabetes causes high blood sugar levels"
        text2 = "High blood sugar is a symptom of diabetes"
        text3 = "Migraine headaches cause severe pain"
        
        emb1 = self.embed_gen.generate_single_embedding(text1)
        emb2 = self.embed_gen.generate_single_embedding(text2)
        emb3 = self.embed_gen.generate_single_embedding(text3)
        
        # Calculate cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        logger.info(f"Similarity (diabetes texts): {sim_12:.3f}")
        logger.info(f"Similarity (diabetes vs migraine): {sim_13:.3f}")
        
        # Similar texts should have higher similarity
        self.assertGreater(sim_12, sim_13,
                          "Similar texts should have higher similarity")
        self.assertGreater(sim_12, 0.5,
                          "Similar texts should have similarity > 0.5")
    
    def test_embedding_dimension(self):
        """Test embedding dimensions"""
        text = "Test medical text"
        embedding = self.embed_gen.generate_single_embedding(text)
        
        self.assertEqual(embedding.shape[0], self.embed_gen.dimension,
                        "Embedding dimension mismatch")


def run_retrieval_tests():
    """Run all retrieval tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingQuality))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_retrieval_tests()
    sys.exit(0 if success else 1)