# Placeholder for document_processor.py
"""
Document processing with structure preservation
"""
from typing import List, Dict, Tuple
import re
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MedicalDocumentProcessor:
    """Process medical documents with section-aware chunking"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = 100
    
    def process_document(self, document: Dict) -> Dict:
        """
        Process a single medical document
        
        Args:
            document: Dict with keys: title, sections, source, url
        
        Returns:
            Dict with processed chunks and metadata
        """
        try:
            # Extract and clean sections
            processed_sections = {}
            for section_name, section_content in document.get('sections', {}).items():
                cleaned_content = self._clean_text(section_content)
                if len(cleaned_content) >= self.min_chunk_length:
                    processed_sections[section_name] = cleaned_content
            
            # Create chunks
            chunks = self._create_section_aware_chunks(
                document.get('title', 'Unknown'),
                processed_sections,
                document
            )
            
            return {
                'document_id': self._generate_doc_id(document),
                'title': document.get('title', 'Unknown'),
                'source': document.get('source', 'Unknown'),
                'url': document.get('url', ''),
                'sections': processed_sections,
                'chunks': chunks,
                'num_chunks': len(chunks),
                'metadata': {
                    'category': document.get('category', 'General'),
                    'credibility_score': document.get('credibility_score', 0.8),
                    'collected_at': document.get('collected_at', '')
                }
            }
        except Exception as e:
            logger.error(f"Error processing document {document.get('title')}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\%\/]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        return text.strip()
    
    def _create_section_aware_chunks(
        self, 
        title: str, 
        sections: Dict[str, str],
        document: Dict
    ) -> List[Dict]:
        """
        Create chunks that preserve section boundaries and context
        """
        chunks = []
        chunk_id = 0
        
        for section_name, section_content in sections.items():
            # Add section context to each chunk
            section_header = f"{title} - {section_name.title()}"
            
            # Split section into chunks
            section_chunks = self._chunk_text(
                section_content,
                self.chunk_size,
                self.chunk_overlap
            )
            
            for chunk_text in section_chunks:
                # Add context header
                full_chunk = f"{section_header}\n\n{chunk_text}"
                
                chunks.append({
                    'chunk_id': f"{self._generate_doc_id(document)}_chunk_{chunk_id}",
                    'text': full_chunk,
                    'section': section_name,
                    'section_header': section_header,
                    'char_count': len(full_chunk),
                    'metadata': {
                        'document_title': title,
                        'section': section_name,
                        'source': document.get('source', 'Unknown'),
                        'url': document.get('url', ''),
                        'chunk_index': chunk_id
                    }
                })
                chunk_id += 1
        
        return chunks
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation point
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclaim = text.rfind('!', start, end)
                
                break_point = max(last_period, last_question, last_exclaim)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_length:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _generate_doc_id(self, document: Dict) -> str:
        """Generate unique document ID"""
        title = document.get('title', 'unknown')
        source = document.get('source', 'unknown')
        
        # Create ID from title and source
        doc_id = f"{source.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}"
        doc_id = re.sub(r'[^\w\-]', '', doc_id)[:100]
        
        return doc_id
    
    def process_batch(self, documents: List[Dict]) -> List[Dict]:
        """Process multiple documents"""
        processed_docs = []
        
        for doc in documents:
            processed = self.process_document(doc)
            if processed:
                processed_docs.append(processed)
        
        logger.info(f"Processed {len(processed_docs)} documents into "
                   f"{sum(d['num_chunks'] for d in processed_docs)} chunks")
        
        return processed_docs
    
    def save_processed_documents(
        self, 
        processed_docs: List[Dict], 
        output_dir: str
    ):
        """Save processed documents"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full processed documents
        with open(output_path / "processed_documents.json", 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)
        
        # Extract all chunks for easy loading
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc['chunks'])
        
        with open(output_path / "all_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(processed_docs)} documents and {len(all_chunks)} chunks to {output_dir}")


if __name__ == "__main__":
    # Test processing
    from src.data_collection.dataset_loader import MedicalDatasetLoader
    
    loader = MedicalDatasetLoader()
    documents = loader.create_sample_medical_dataset()
    
    processor = MedicalDocumentProcessor(chunk_size=512, chunk_overlap=50)
    processed = processor.process_batch(documents)
    processor.save_processed_documents(processed, "./data/processed")
    
    print(f"Processed {len(processed)} documents")
    print(f"Total chunks: {sum(d['num_chunks'] for d in processed)}")