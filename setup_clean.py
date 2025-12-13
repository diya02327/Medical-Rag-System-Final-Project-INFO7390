"""
Clean setup with manual embedding management
"""
import os
import sys
import json
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*70)
print("MEDICAL RAG - CLEAN SETUP")
print("="*70)

# Create directories
for d in ["data/raw", "data/processed", "data/vector_db/chroma", "logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Sample data
sample_data = [
    {
        "title": "Type 2 Diabetes",
        "sections": {
            "overview": "Type 2 diabetes is a chronic condition affecting blood sugar metabolism. The body resists insulin or doesn't produce enough insulin.",
            "symptoms": "Symptoms include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, and slow-healing sores.",
            "treatment": "Treatment includes lifestyle changes like healthy eating and exercise, along with medications to manage blood sugar levels."
        }
    },
    {
        "title": "Migraine Headaches",
        "sections": {
            "overview": "A migraine causes severe throbbing pain, usually on one side of the head, often with nausea and light sensitivity.",
            "symptoms": "Symptoms include severe head pain, nausea, vomiting, sensitivity to light and sound, and visual disturbances.",
            "treatment": "Treatment includes pain-relieving medications, preventive medications, and lifestyle changes to avoid triggers."
        }
    },
    {
        "title": "Hypertension",
        "sections": {
            "overview": "High blood pressure occurs when blood force against artery walls is consistently too high. Often has no symptoms.",
            "symptoms": "Usually no symptoms. Severe cases may cause headaches, shortness of breath, or nosebleeds.",
            "prevention": "Prevention includes healthy diet, regular exercise, maintaining healthy weight, limiting alcohol, and managing stress."
        }
    },
    {
        "title": "Asthma",
        "sections": {
            "overview": "Asthma causes airways to narrow and swell, making breathing difficult. Can trigger coughing and wheezing.",
            "symptoms": "Symptoms include shortness of breath, chest tightness, wheezing when exhaling, and trouble sleeping due to breathing issues.",
            "treatment": "Treatment includes inhaled corticosteroids for long-term control and quick-relief inhalers for acute symptoms."
        }
    },
    {
        "title": "Anxiety Disorders",
        "sections": {
            "overview": "Anxiety disorders involve persistent excessive worry and fear about everyday situations that interfere with daily activities.",
            "symptoms": "Symptoms include nervousness, sense of danger, increased heart rate, rapid breathing, sweating, trembling, and trouble concentrating.",
            "treatment": "Treatment includes cognitive behavioral therapy, medications like antidepressants, and lifestyle changes."
        }
    }
]

# Save
with open("data/raw/medical_data.json", 'w') as f:
    json.dump(sample_data, f, indent=2)
logger.info("✅ Created sample data")

# Create chunks
chunks = []
for doc in sample_data:
    for section, content in doc['sections'].items():
        chunks.append({
            'id': f"{doc['title'].lower().replace(' ', '_')}_{section}",
            'text': f"{doc['title']} - {section.title()}\n\n{content}",
            'metadata': {
                'title': doc['title'],
                'section': section
            }
        })

with open("data/processed/chunks.json", 'w') as f:
    json.dump(chunks, f, indent=2)
logger.info(f"✅ Created {len(chunks)} chunks")

# Generate embeddings manually
print("\n📊 Generating embeddings...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [c['text'] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

logger.info(f"✅ Generated embeddings: {embeddings.shape}")

# Save embeddings
np.save("data/processed/embeddings.npy", embeddings)

# Store in ChromaDB WITHOUT using its embedding function
print("\n🗄️  Setting up ChromaDB...")
import chromadb

client = chromadb.PersistentClient(
    path="data/vector_db/chroma",
    settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
)

# Delete old
try:
    client.delete_collection("medical_knowledge")
except:
    pass

# Create WITHOUT embedding function
collection = client.create_collection(
    name="medical_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Add with pre-computed embeddings
for i, chunk in enumerate(chunks):
    collection.add(
        embeddings=[embeddings[i].tolist()],  # Use our embeddings
        documents=[chunk['text']],
        metadatas=[chunk['metadata']],
        ids=[chunk['id']]
    )

logger.info(f"✅ Added {len(chunks)} documents to ChromaDB")

# Test
print("\n🔍 Testing...")
# Query by providing embedding
test_query = "What are diabetes symptoms?"
test_emb = model.encode([test_query])[0]

results = collection.query(
    query_embeddings=[test_emb.tolist()],
    n_results=3
)

logger.info(f"✅ Query works! Found {len(results['documents'][0])} results")

print("\n" + "="*70)
print("✅ SETUP COMPLETE!")
print("="*70)
print("\n🚀 Run: streamlit run app_working.py")