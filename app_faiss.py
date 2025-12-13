"""
Medical RAG Assistant with FAISS - No ChromaDB issues!
"""
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize system (cached)
@st.cache_resource
def load_system():
    # Load FAISS index
    index = faiss.read_index("data/vector_db/faiss.index")
    
    # Load chunks
    with open("data/vector_db/chunks.pkl", 'rb') as f:
        chunks = pickle.load(f)
    
    # Load embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    return index, chunks, embed_model, openai_client

# Load
try:
    index, chunks, embed_model, openai_client = load_system()
    st.success(f"✅ System ready with {len(chunks)} medical documents")
except FileNotFoundError:
    st.error("❌ Database not found!")
    st.info("Run: **python setup_faiss.py**")
    st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Main UI
st.title("🏥 Medical Information Assistant")

st.warning("""
⚠️ **Medical Disclaimer**: This tool provides general medical information for educational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with a qualified healthcare provider for medical concerns.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    n_results = st.slider("Number of sources", 3, 10, 5)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.3)
    
    st.markdown("---")
    
    st.header("📊 Statistics")
    st.metric("Documents Indexed", len(chunks))
    st.metric("Queries This Session", len(st.session_state.get('messages', [])) // 2)
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.info("""
    Provides evidence-based medical information from reputable sources.
    
    **Technology:**
    - FAISS vector search
    - Sentence transformers
    - OpenAI GPT-4
    """)
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Example questions
st.subheader("💡 Try These Questions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What are diabetes symptoms?", use_container_width=True):
        st.session_state.next_query = "What are the symptoms of type 2 diabetes?"

with col2:
    if st.button("How to prevent hypertension?", use_container_width=True):
        st.session_state.next_query = "How can I prevent high blood pressure?"

with col3:
    if st.button("What triggers asthma?", use_container_width=True):
        st.session_state.next_query = "What are common asthma triggers?"

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source['title']} - {source['section']}**")
                    st.caption(source['text'][:200] + "...")
                    st.markdown("---")

# Get query
if "next_query" in st.session_state:
    query = st.session_state.next_query
    del st.session_state.next_query
else:
    query = st.chat_input("Ask a medical question...")

# Process query
if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching medical database..."):
            try:
                # Generate query embedding
                query_embedding = embed_model.encode([query]).astype('float32')
                
                # Search FAISS
                distances, indices = index.search(query_embedding, n_results)
                
                # Get results
                sources = []
                context_parts = []
                
                for i, idx in enumerate(indices[0]):
                    chunk = chunks[idx]
                    sources.append({
                        'title': chunk['metadata']['title'],
                        'section': chunk['metadata']['section'],
                        'text': chunk['text']
                    })
                    context_parts.append(f"[Source {i+1}]\n{chunk['text']}")
                
                context = "\n\n".join(context_parts)
                
                # Generate response with OpenAI
                with st.spinner("🤖 Generating response..."):
                    completion = openai_client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": """You are a medical information assistant. Provide accurate, 
                                evidence-based information from the context. Always cite sources. 
                                Include a medical disclaimer. Never diagnose conditions."""
                            },
                            {
                                "role": "user",
                                "content": f"""Based on these medical sources:

{context}

User Question: {query}

Provide a clear, helpful answer with citations. Remind user to consult healthcare professionals."""
                            }
                        ],
                        temperature=temperature,
                        max_tokens=1200
                    )
                    
                    answer = completion.choices[0].message.content
                
                # Display answer
                st.markdown(answer)
                
                # Disclaimer
                st.markdown("---")
                st.caption("⚠️ This information is for educational purposes only. Always consult qualified healthcare providers.")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Show sources
                with st.expander(f"📚 {len(sources)} Sources Used"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}. {source['title']}** - *{source['section'].title()}*")
                        st.text(source['text'][:250] + "...")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
Medical Information Assistant | FAISS + Sentence Transformers + OpenAI GPT-4<br>
Always consult healthcare professionals for medical advice
</div>
""", unsafe_allow_html=True)