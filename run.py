# Placeholder for run.py
"""
Main script to build and run the Medical Information Assistant
"""
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import modules
from src.data_collection.dataset_loader import MedicalDatasetLoader
from src.preprocessing.document_processor import MedicalDocumentProcessor
from src.vector_db.chroma_manager import ChromaDBManager
from src.vector_db.faiss_manager import FAISSManager
from src.vector_db.embeddings import MedicalEmbeddingGenerator


def setup_data():
    """Setup: Create sample medical dataset"""
    logger.info("=== Step 1: Setting up medical dataset ===")
    
    loader = MedicalDatasetLoader()
    documents = loader.create_sample_medical_dataset()
    
    # Also load any existing documents
    all_documents = loader.load_all_datasets()
    
    logger.info(f"Loaded {len(all_documents)} medical documents")
    return all_documents


def process_documents(documents):
    """Process documents into chunks"""
    logger.info("=== Step 2: Processing documents ===")
    
    processor = MedicalDocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    
    processed_docs = processor.process_batch(documents)
    processor.save_processed_documents(
        processed_docs,
        os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    )
    
    logger.info(f"Processed {len(processed_docs)} documents")
    return processed_docs


def build_vector_databases(processed_docs):
    """Build ChromaDB and FAISS vector databases"""
    logger.info("=== Step 3: Building vector databases ===")
    
    # Extract all chunks
    all_chunks = []
    for doc in processed_docs:
        all_chunks.extend(doc['chunks'])
    
    logger.info(f"Total chunks to index: {len(all_chunks)}")
    
    # Initialize embedding generator
    embed_gen = MedicalEmbeddingGenerator(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Prepare data for indexing
    documents = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]
    ids = [chunk['chunk_id'] for chunk in all_chunks]
    
    # Build ChromaDB
    logger.info("Building ChromaDB...")
    chroma = ChromaDBManager(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/vector_db/chroma")
    )
    chroma.create_collection(reset=True)
    chroma.add_documents(documents, metadatas, ids)
    logger.info(f"ChromaDB stats: {chroma.get_collection_stats()}")
    
    # Build FAISS
    logger.info("Building FAISS index...")
    faiss_mgr = FAISSManager(
        dimension=embed_gen.dimension,
        persist_directory=os.getenv("FAISS_PERSIST_DIR", "./data/vector_db/faiss")
    )
    
    # Generate embeddings
    embeddings = embed_gen.generate_embeddings(documents, batch_size=32)
    faiss_mgr.add_documents(embeddings, documents, metadatas, ids)
    faiss_mgr.save()
    logger.info(f"FAISS stats: {faiss_mgr.get_stats()}")
    
    return chroma, faiss_mgr


def run_streamlit_app():
    """Launch Streamlit application"""
    logger.info("=== Step 4: Launching Streamlit app ===")
    
    import subprocess
    
    app_path = Path(__file__).parent / "src" / "ui" / "app.py"
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error running Streamlit: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Medical Information Assistant")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup and build vector databases"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run Streamlit application"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full setup and run"
    )
    
    args = parser.parse_args()
    
    try:
        if args.setup or args.full:
            # Setup data
            documents = setup_data()
            
            # Process documents
            processed_docs = process_documents(documents)
            
            # Build vector databases
            build_vector_databases(processed_docs)
            
            logger.info("✅ Setup complete!")
        
        if args.run or args.full:
            # Run Streamlit app
            run_streamlit_app()
        
        if not (args.setup or args.run or args.full):
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()