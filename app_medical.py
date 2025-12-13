"""
Medical Information Assistant - Clean Version with FAISS
NO ChromaDB, NO Proxy Issues
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

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize (cached to avoid reloading)
@st.cache_resource
def initialize_system():
    """Load all system components"""
    try:
        # Load FAISS index
        index = faiss.read_index("data/vector_db/faiss_index.bin")
        
        # Load chunks
        with open("data/vector_db/chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        # Load embedding model
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI - NO PROXIES
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        return {
            'index': index,
            'chunks': chunks,
            'embed_model': embed_model,
            'openai_client': openai_client,
            'status': 'success'
        }
    except FileNotFoundError as e:
        return {'status': 'error', 'message': f'Database not found: {e}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Load system
system = initialize_system()

if system['status'] == 'error':
    st.error(f"❌ Error: {system['message']}")
    st.info("**To fix:** Run `python setup_faiss.py` first")
    st.stop()

# Unpack components
index = system['index']
chunks = system['chunks']
embed_model = system['embed_model']
openai_client = system['openai_client']

st.success(f"✅ System ready: {len(chunks)} medical documents indexed")

# Main UI
st.title("🏥 Medical Information Assistant")

# Medical Disclaimer
st.warning("""
⚠️ **MEDICAL DISCLAIMER**

This tool provides **general medical information for educational purposes only**. 

**This is NOT:**
- Medical advice
- A diagnosis tool
- A substitute for professional medical care

**Always:**
- Consult with qualified healthcare providers for medical concerns
- Seek emergency care for urgent symptoms
- Discuss any health changes with your doctor
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    n_results = st.slider(
        "Number of sources to retrieve",
        min_value=3,
        max_value=10,
        value=5,
        help="More sources = more comprehensive but potentially less focused"
    )
    
    temperature = st.slider(
        "Response creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.markdown("---")
    
    st.header("📊 Database Stats")
    st.metric("Documents Indexed", len(chunks))
    st.metric("Vector Dimension", index.d)
    st.metric("Queries This Session", len(st.session_state.get('messages', [])) // 2)
    
    st.markdown("---")
    
    st.header("ℹ️ Technology")
    st.info("""
    **Vector Search:** FAISS  
    **Embeddings:** Sentence Transformers  
    **LLM:** OpenAI GPT-4  
    **Sources:** Medical Knowledge Base
    """)
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Example questions
st.subheader("💡 Example Questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What are diabetes symptoms?", use_container_width=True):
        st.session_state.pending_query = "What are the symptoms of type 2 diabetes?"
        st.rerun()

with col2:
    if st.button("How to prevent hypertension?", use_container_width=True):
        st.session_state.pending_query = "How can I prevent high blood pressure?"
        st.rerun()

with col3:
    if st.button("What triggers asthma attacks?", use_container_width=True):
        st.session_state.pending_query = "What are common asthma triggers?"
        st.rerun()

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View {len(message['sources'])} Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source['title']}** - *{source['section'].title()}*")
                    st.caption(source['text'][:200] + "...")
                    st.markdown("---")

# Get user input
query = None
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
else:
    query = st.chat_input("Ask a medical question... (e.g., 'What are migraine symptoms?')")

# Process query
if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Step 1: Search
            with st.spinner("🔍 Searching medical database..."):
                query_embedding = embed_model.encode([query]).astype('float32')
                distances, indices = index.search(query_embedding, n_results)
            
            # Step 2: Collect sources
            sources = []
            context_parts = []
            
            for i, idx in enumerate(indices[0]):
                chunk = chunks[idx]
                sources.append({
                    'title': chunk['metadata']['title'],
                    'section': chunk['metadata']['section'],
                    'text': chunk['text'],
                    'distance': float(distances[0][i])
                })
                context_parts.append(f"[Source {i+1}: {chunk['metadata']['title']}]\n{chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Generate with OpenAI
            with st.spinner("🤖 Generating response..."):
                response = openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a medical information assistant. 

Your role:
- Provide accurate, evidence-based information from the provided context
- Use clear, patient-friendly language
- Cite sources using [Source X] format
- Never diagnose conditions or provide specific medical advice
- Always remind users to consult healthcare professionals

Guidelines:
- Base answers ONLY on the provided context
- If context doesn't contain the answer, say so clearly
- Be empathetic but factual
- Include relevant warnings when appropriate"""
                        },
                        {
                            "role": "user",
                            "content": f"""Medical Sources:

{context}

User Question: {query}

Provide a comprehensive, evidence-based answer that:
1. Directly addresses the question
2. Cites specific sources [Source 1], [Source 2], etc.
3. Uses clear, accessible language
4. Includes relevant precautions or warnings
5. Ends with a reminder to consult healthcare professionals"""
                        }
                    ],
                    temperature=temperature,
                    max_tokens=1500
                )
                
                answer = response.choices[0].message.content
            
            # Display answer
            st.markdown(answer)
            
            # Medical disclaimer
            st.markdown("---")
            st.caption("""
⚠️ **Remember:** This information is for educational purposes only. 
Always consult with qualified healthcare providers for medical advice, diagnosis, or treatment.
If you have a medical emergency, call emergency services immediately.
            """)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
            # Show sources
            with st.expander(f"📚 View {len(sources)} Sources Used"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}: {source['title']}** - *{source['section'].title()}*")
                    st.markdown(f"*Relevance: {1 / (1 + source['distance']):.2%}*")
                    st.text_area(
                        f"Content {i}",
                        source['text'],
                        height=150,
                        disabled=True,
                        key=f"source_{i}_{len(st.session_state.messages)}"
                    )
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em; padding: 20px;'>
    <strong>Medical Information Assistant</strong><br>
    FAISS Vector Search • Sentence Transformers • OpenAI GPT-4<br>
    <br>
    <em>For educational purposes only • Always consult healthcare professionals</em>
</div>
""", unsafe_allow_html=True)