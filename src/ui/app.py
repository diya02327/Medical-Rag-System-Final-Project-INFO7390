# Placeholder for app.py
"""
Medical Information Assistant - Streamlit UI
"""
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vector_db.chroma_manager import ChromaDBManager
from src.llm.rag_pipeline import MedicalRAGPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #28a745;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 3px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline (cached)"""
    try:
        # Initialize ChromaDB
        chroma = ChromaDBManager(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/vector_db/chroma"),
            collection_name="medical_knowledge"
        )
        chroma.create_collection()
        
        # Initialize RAG pipeline
        rag = MedicalRAGPipeline(
            vector_db_manager=chroma,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1500"))
        )
        
        return rag, chroma
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        logger.error(f"Initialization error: {e}")
        return None, None


def display_disclaimer():
    """Display medical disclaimer"""
    st.markdown("""
    <div class="disclaimer-box">
        <h4>⚠️ Important Medical Disclaimer</h4>
        <p>This tool provides general medical information for educational purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with a qualified healthcare provider for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True)


def display_sources(sources: list):
    """Display source information"""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source['source']} - {source['title']}", expanded=False):
            st.markdown(f"**Section:** {source['section'].title()}")
            st.markdown(f"**Relevance Score:** {source['relevance_score']:.2%}")
            st.text_area(
                "Content Preview",
                source['text'],
                height=150,
                key=f"source_{i}",
                disabled=True
            )
            if source.get('url'):
                st.markdown(f"[View Original Source]({source['url']})")


def display_questions_for_doctor(rag_pipeline, condition: str):
    """Display questions to ask doctor"""
    with st.spinner("Generating questions to ask your doctor..."):
        questions = rag_pipeline.generate_questions_for_doctor(condition)
        
        if questions:
            st.subheader("💬 Questions to Ask Your Doctor")
            for i, question in enumerate(questions, 1):
                st.markdown(f"{i}. {question}")


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Information Assistant</h1>', unsafe_allow_html=True)
    
    # Display disclaimer
    display_disclaimer()
    
    # Initialize RAG pipeline
    rag_pipeline, chroma = initialize_rag_pipeline()
    
    if not rag_pipeline:
        st.error("Failed to initialize the application. Please check your configuration.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Number of sources
        n_sources = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources provide more context but may include less relevant information"
        )
        
        # Query type
        query_type = st.selectbox(
            "Query Type",
            ["auto", "general", "symptoms", "condition"],
            help="Auto-detect or manually specify query type"
        )
        
        st.markdown("---")
        
        # Statistics
        st.header("📊 Statistics")
        try:
            stats = chroma.get_collection_stats()
            st.metric("Documents Indexed", stats['document_count'])
            st.metric("Queries This Session", len(st.session_state.get('chat_history', [])) // 2)
        except:
            st.info("Statistics unavailable")
        
        st.markdown("---")
        
        # Information
        st.header("ℹ️ About")
        st.info("""
        This Medical Information Assistant provides evidence-based information from reputable medical sources including:
        
        - MedlinePlus
        - Mayo Clinic
        - CDC
        - Other peer-reviewed sources
        
        **Features:**
        - Semantic search through medical literature
        - Evidence-based responses
        - Source citations
        - Questions to ask your doctor
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Example queries
    st.subheader("💡 Example Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What are the symptoms of type 2 diabetes?", use_container_width=True):
            st.session_state.example_query = "What are the symptoms of type 2 diabetes?"
    
    with col2:
        if st.button("How can I prevent high blood pressure?", use_container_width=True):
            st.session_state.example_query = "How can I prevent high blood pressure?"
    
    with col3:
        if st.button("What causes migraine headaches?", use_container_width=True):
            st.session_state.example_query = "What causes migraine headaches?"
    
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🏥 **Assistant:** {message["content"]}</div>', 
                           unsafe_allow_html=True)
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    display_sources(message["sources"])
    
    # Query input
    query_input = st.chat_input("Ask a medical question...")
    
    # Handle example query
    if 'example_query' in st.session_state:
        query_input = st.session_state.example_query
        del st.session_state.example_query
    
    # Process query
    if query_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query_input
        })
        
        # Display user message
        with chat_container:
            st.markdown(f'<div class="chat-message user-message">👤 **You:** {query_input}</div>', 
                       unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🔍 Searching medical knowledge base and generating response..."):
            try:
                # Determine query type
                final_query_type = "general" if query_type == "auto" else query_type
                
                # Query RAG pipeline
                result = rag_pipeline.query(
                    user_query=query_input,
                    n_results=n_sources,
                    chat_history=st.session_state.chat_history[:-1],  # Exclude current query
                    query_type=final_query_type
                )
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources'],
                    "query_type": result['query_type']
                })
                
                # Display response
                with chat_container:
                    st.markdown(f'<div class="chat-message assistant-message">🏥 **Assistant:** {result["answer"]}</div>', 
                               unsafe_allow_html=True)
                    
                    # Display sources
                    display_sources(result['sources'])
                    
                    # Generate questions for doctor if it's a symptom or condition query
                    if result['query_type'] in ['symptoms', 'condition']:
                        display_questions_for_doctor(rag_pipeline, query_input)
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                logger.error(f"Query error: {e}")
        
        # Rerun to update display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Medical Information Assistant | Powered by RAG & OpenAI GPT-4</p>
        <p>Always consult healthcare professionals for medical advice</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()