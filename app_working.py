"""
Working Medical RAG Assistant
"""
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medical Assistant", page_icon="🏥", layout="wide")

# Initialize (cached)
@st.cache_resource
def init_system():
    # Embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # ChromaDB
    client = chromadb.PersistentClient(
        path="data/vector_db/chroma",
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection("medical_knowledge")
    
    # OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    return embed_model, collection, openai_client

try:
    embed_model, collection, openai_client = init_system()
    st.success("✅ System ready")
except Exception as e:
    st.error(f"Setup error: {e}")
    st.info("Run: python setup_clean.py")
    st.stop()

# UI
st.title("🏥 Medical Information Assistant")

st.warning("""
⚠️ **Disclaimer**: Educational purposes only. NOT medical advice. 
Always consult healthcare professionals.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    n_results = st.slider("Sources", 3, 10, 5)
    
    st.markdown("---")
    st.metric("Documents", collection.count())
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Examples
st.subheader("💡 Try These")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Diabetes symptoms?"):
        st.session_state.next_q = "What are the symptoms of type 2 diabetes?"

with col2:
    if st.button("Prevent high BP?"):
        st.session_state.next_q = "How to prevent hypertension?"

with col3:
    if st.button("Asthma triggers?"):
        st.session_state.next_q = "What triggers asthma attacks?"

st.markdown("---")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Get input
if "next_q" in st.session_state:
    query = st.session_state.next_q
    del st.session_state.next_q
else:
    query = st.chat_input("Ask a medical question...")

# Process
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # Get query embedding
            query_emb = embed_model.encode([query])[0]
            
            # Search ChromaDB with our embedding
            results = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=n_results
            )
            
            # Build context
            context = "\n\n".join([
                f"[{i+1}] {doc}" 
                for i, doc in enumerate(results['documents'][0])
            ])
            
            # Generate response
            with st.spinner("Generating..."):
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a medical information assistant. Use the provided context to answer questions. Include disclaimer."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a helpful answer with sources."}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                
                answer = response.choices[0].message.content
            
            st.write(answer)
            
            st.caption("⚠️ This is educational information only. Consult healthcare providers for medical advice.")
            
            # Show sources
            with st.expander("📚 Sources"):
                for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                    st.markdown(f"**{i}. {meta['title']} - {meta['section']}**")
                    st.caption(doc[:150] + "...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption("Medical Assistant | ChromaDB + OpenAI")