# Placeholder for rag_pipeline.py
"""
RAG Pipeline for Medical Information Assistant
"""
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
import logging
import re
from src.llm.prompts import (
    SYSTEM_PROMPT, 
    USER_QUERY_TEMPLATE, 
    SYMPTOM_QUERY_TEMPLATE,
    CONDITION_INFO_TEMPLATE,
    DISCLAIMER,
    get_prompt_template
)

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """RAG pipeline for medical information retrieval and generation"""
    
    def __init__(
        self,
        vector_db_manager,
        openai_api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.3,
        max_tokens: int = 1500
    ):
        self.vector_db = vector_db_manager
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Medical RAG Pipeline initialized with model: {model}")
    
    def query(
        self,
        user_query: str,
        n_results: int = 5,
        chat_history: Optional[List[Dict]] = None,
        query_type: str = "general"
    ) -> Dict:
        """
        Process user query through RAG pipeline
        
        Args:
            user_query: User's question
            n_results: Number of documents to retrieve
            chat_history: Previous conversation history
            query_type: Type of query (general, symptoms, condition, followup)
        
        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            # Step 1: Classify query intent
            detected_type = self._classify_query(user_query)
            if query_type == "general":
                query_type = detected_type
            
            # Step 2: Retrieve relevant documents
            logger.info(f"Retrieving documents for query: {user_query}")
            retrieved_docs = self.vector_db.query(user_query, n_results=n_results)
            
            if not retrieved_docs['documents']:
                return self._generate_no_results_response(user_query)
            
            # Step 3: Build context from retrieved documents
            context = self._build_context(retrieved_docs)
            
            # Step 4: Check if context is relevant
            if not self._is_context_relevant(user_query, context):
                return self._generate_no_results_response(user_query)
            
            # Step 5: Generate response
            response = self._generate_response(
                user_query, 
                context, 
                chat_history,
                query_type
            )
            
            # Step 6: Add medical disclaimer
            response_with_disclaimer = f"{response}\n\n{DISCLAIMER}"
            
            return {
                'answer': response_with_disclaimer,
                'sources': self._format_sources(retrieved_docs),
                'context': context,
                'query_type': query_type,
                'retrieved_count': len(retrieved_docs['documents'])
            }
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'answer': f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again.",
                'sources': [],
                'context': "",
                'query_type': query_type,
                'retrieved_count': 0
            }
    
    def _classify_query(self, query: str) -> str:
        """Classify query type based on content"""
        query_lower = query.lower()
        
        # Symptom-related keywords
        symptom_keywords = [
            'symptom', 'feel', 'experiencing', 'having', 'pain', 'ache',
            'hurt', 'discomfort', 'notice', 'suffer', 'problem with'
        ]
        
        # Condition information keywords
        condition_keywords = [
            'what is', 'tell me about', 'explain', 'information about',
            'learn about', 'understand', 'how does', 'causes of'
        ]
        
        if any(keyword in query_lower for keyword in symptom_keywords):
            return "symptoms"
        elif any(keyword in query_lower for keyword in condition_keywords):
            return "condition"
        else:
            return "general"
    
    def _build_context(self, retrieved_docs: Dict) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        for i, (doc, metadata) in enumerate(zip(
            retrieved_docs['documents'], 
            retrieved_docs['metadatas']
        ), 1):
            source = metadata.get('source', 'Medical Source')
            section = metadata.get('section', 'General')
            
            context_part = f"""
[Source {i}: {source} - {section.title()}]
{doc}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _is_context_relevant(self, query: str, context: str) -> bool:
        """Check if retrieved context is relevant to query"""
        # Simple relevance check - can be enhanced with embedding similarity
        query_terms = set(query.lower().split())
        context_lower = context.lower()
        
        # Check if at least 20% of query terms appear in context
        matching_terms = sum(1 for term in query_terms if term in context_lower)
        relevance_ratio = matching_terms / len(query_terms) if query_terms else 0
        
        return relevance_ratio >= 0.2
    
    def _generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]],
        query_type: str
    ) -> str:
        """Generate response using LLM"""
        
        # Select appropriate prompt template
        prompt_template = get_prompt_template(query_type)
        
        # Build user message
        if chat_history and len(chat_history) > 0:
            # Include chat history for context
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in chat_history[-4:]  # Last 4 messages
            ])
            user_message = prompt_template.format(
                context=context,
                query=query,
                chat_history=history_text
            )
        else:
            user_message = prompt_template.format(
                context=context,
                query=query
            )
        
        # Call OpenAI API
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        logger.info(f"Generating response with {self.model}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def _format_sources(self, retrieved_docs: Dict) -> List[Dict]:
        """Format sources for display"""
        sources = []
        
        for doc, metadata, distance in zip(
            retrieved_docs['documents'],
            retrieved_docs['metadatas'],
            retrieved_docs['distances']
        ):
            sources.append({
                'text': doc[:300] + "..." if len(doc) > 300 else doc,
                'source': metadata.get('source', 'Unknown'),
                'section': metadata.get('section', 'General'),
                'title': metadata.get('document_title', 'Medical Information'),
                'url': metadata.get('url', ''),
                'relevance_score': float(1 - distance) if distance else 0.0
            })
        
        return sources
    
    def _generate_no_results_response(self, query: str) -> Dict:
        """Generate response when no relevant documents found"""
        response = f"""I apologize, but I don't have specific information about "{query}" in my current medical knowledge base. 

This could mean:
1. The topic might be outside the scope of my current knowledge
2. It might be a very specialized medical topic
3. The question might need to be rephrased

I strongly recommend:
- Consulting with a healthcare professional who can provide personalized advice
- Checking reputable medical websites like Mayo Clinic, MedlinePlus, or CDC
- Scheduling an appointment with your doctor if this is related to your health

{DISCLAIMER}
"""
        return {
            'answer': response,
            'sources': [],
            'context': "",
            'query_type': "no_results",
            'retrieved_count': 0
        }
    
    def generate_questions_for_doctor(
        self,
        condition: str,
        user_symptoms: Optional[str] = None
    ) -> List[str]:
        """Generate relevant questions to ask a doctor"""
        
        prompt = f"""Based on the medical condition "{condition}" {f'and symptoms: {user_symptoms}' if user_symptoms else ''}, generate 8-10 specific, relevant questions that a patient should ask their doctor during a consultation.

Questions should cover:
1. Diagnosis confirmation
2. Treatment options
3. Prognosis and timeline
4. Lifestyle modifications
5. Warning signs
6. Follow-up care
7. Medication details (if applicable)

Format as a numbered list."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant that helps patients prepare for doctor visits."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )
        
        questions_text = response.choices[0].message.content
        
        # Extract numbered list
        questions = re.findall(r'\d+\.\s*(.+)', questions_text)
        return questions[:10]


if __name__ == "__main__":
    # Test RAG pipeline
    from src.vector_db.chroma_manager import ChromaDBManager
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize components
    chroma = ChromaDBManager()
    chroma.create_collection()
    
    # Initialize RAG pipeline
    rag = MedicalRAGPipeline(
        vector_db_manager=chroma,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    result = rag.query(test_query, n_results=3)
    
    print("Answer:", result['answer'])
    print("\nSources:", len(result['sources']))