"""
Simplified Medical Information Assistant - Fixed OpenAI version
"""
import streamlit as st
import os
from pathlib import Path
import sys
import traceback

# Add to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config MUST be first
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Information Assistant")
st.write("Loading system...")

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API Key not found!")
    st.info("Add OPENAI_API_KEY to your .env file")
    st.stop()

st.success("✅ API Key loaded")

# Import libraries
try:
    st.write("Loading libraries...")
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
    st.success("✅ Libraries loaded")
except Exception as e:
    st.error(f"Error loading libraries: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Initialize ChromaDB
try:
    st.write("Connecting to database...")
    chroma_dir = Path("data/vector_db/chroma")
    
    if not chroma_dir.exists():
        st.error(f"❌ Database not found at: {chroma_dir}")
        st.info("Run: python setup_system.py")
        st.stop()
    
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection(name="medical_knowledge")
    doc_count = collection.count()
    
    st.success(f"✅ Database connected: {doc_count} documents")
    
except Exception as e:
    st.error(f"Database error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Initialize OpenAI - FIXED VERSION
try:
    st.write("Connecting to OpenAI...")
    
    # Simple initialization without extra parameters
    openai_client = OpenAI(api_key=api_key)
    
    st.success("✅ OpenAI connected")
    
except Exception as e:
    st.error(f"OpenAI error: {e}")
    st.code(traceback.format_exc())
    st.info("Make sure you have openai==1.12.0 installed")
    st.stop()

# Clear loading messages
for _ in range(10):
    st.empty()

# Main App UI
st.title("🏥 Medical Information Assistant")

st.warning("""
⚠️ **Medical Disclaimer**: This provides general information for educational purposes only. 
NOT a substitute for professional medical advice. Always consult healthcare providers.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    n_sources = st.slider("Sources to retrieve", 3, 10, 5)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.3)
    
    st.markdown("---")
    st.header("📊 Stats")
    st.metric("Documents", doc_count)
    st.metric("Queries", len(st.session_state.get("messages", [])) // 2)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example questions
st.subheader("💡 Example Questions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What are diabetes symptoms?", use_container_width=True):
        st.session_state.next_query = "What are the symptoms of type 2 diabetes?"

with col2:
    if st.button("How to prevent high BP?", use_container_width=True):
        st.session_state.next_query = "How can I prevent high blood pressure?"

with col3:
    if st.button("What causes migraines?", use_container_width=True):
        st.session_state.next_query = "What causes migraine headaches?"

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['title']} ({src['section']})**")
                    st.caption(src['text'][:150] + "...")

# Get user input
if "next_query" in st.session_state:
    user_query = st.session_state.next_query
    del st.session_state.next_query
else:
    user_query = st.chat_input("Ask a medical question...")

# Process query
if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Retrieve documents
            with st.spinner("🔍 Searching medical database..."):
                results = collection.query(
                    query_texts=[user_query],
                    n_results=n_sources
                )
            
            if not results['documents'][0]:
                st.warning("No relevant information found.")
                st.stop()
            
            # Build context and sources
            context_parts = []
            sources = []
            
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                context_parts.append(f"[Source {i}: {meta.get('document_title', 'Unknown')}]\n{doc}\n")
                sources.append({
                    'title': meta.get('document_title', 'Unknown'),
                    'section': meta.get('section', 'general'),
                    'text': doc
                })
            
            context = "\n".join(context_parts)
            
            # Generate response
            with st.spinner("🤖 Generating response..."):
                completion = openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical information assistant. Provide accurate, evidence-based information from the context provided. Cite sources. Include medical disclaimer. Never diagnose."
                        },
                        {
                            "role": "user",
                            "content": f"""Based on these medical sources:

{context}

Question: {user_query}

Provide a clear answer with citations. Remind user to consult healthcare professionals."""
                        }
                    ],
                    temperature=temperature,
                    max_tokens=1200
                )
                
                answer = completion.choices[0].message.content
            
            # Display answer
            st.markdown(answer)
            
            # Medical disclaimer
            st.markdown("---")
            st.caption("⚠️ This information is for educational purposes only. Always consult qualified healthcare providers for medical advice.")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
            # Show sources
            with st.expander(f"📚 {len(sources)} Sources Used"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}. {src['title']}** - *{src['section'].title()}*")
                    st.text(src['text'][:200] + "...")
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
Medical Information Assistant | ChromaDB + OpenAI GPT-4<br>
Always consult healthcare professionals for medical advice
</div>
""", unsafe_allow_html=True)