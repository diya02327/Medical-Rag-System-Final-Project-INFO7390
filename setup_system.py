"""
Complete system setup - Fixed for compatibility
"""
import os
import sys
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*70)
print("MEDICAL RAG ASSISTANT - COMPLETE SETUP")
print("="*70)

# Step 0: Create directories
print("\n📁 Creating directories...")
directories = [
    "data/raw",
    "data/processed",
    "data/vector_db/chroma",
    "logs"
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created: {directory}")

# Step 1: Create sample medical data
print("\n📦 Step 1: Creating sample medical data...")

sample_medical_data = [
    {
        "source": "Medical Knowledge Base",
        "title": "Type 2 Diabetes",
        "category": "Endocrine Disorders",
        "sections": {
            "overview": "Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar (glucose). With type 2 diabetes, your body either resists the effects of insulin or doesn't produce enough insulin to maintain normal glucose levels. This is the most common form of diabetes, affecting millions of people worldwide.",
            "symptoms": "Common symptoms include increased thirst and frequent urination, increased hunger, unintended weight loss, fatigue and weakness, blurred vision, slow-healing sores or frequent infections, numbness or tingling in hands or feet, and areas of darkened skin, usually in the armpits and neck.",
            "causes": "Type 2 diabetes develops when the body becomes resistant to insulin or when the pancreas is unable to produce enough insulin. Excess weight is strongly linked to the development of type 2 diabetes. Risk factors include weight, inactivity, family history, age, gestational diabetes, and polycystic ovary syndrome.",
            "treatment": "Treatment includes lifestyle changes such as healthy eating and exercise, along with medications to help manage blood sugar levels. Some people may need insulin therapy.",
            "prevention": "Healthy lifestyle choices can help prevent type 2 diabetes. Key prevention strategies include eating healthy foods, getting active with at least 30 minutes of moderate physical activity daily, losing excess weight, and avoiding sedentary behaviors."
        },
        "url": "https://www.mayoclinic.org/diseases-conditions/type-2-diabetes"
    },
    {
        "source": "Medical Knowledge Base",
        "title": "Migraine Headaches",
        "category": "Neurological Disorders",
        "sections": {
            "overview": "A migraine is a type of headache characterized by severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days.",
            "symptoms": "Common symptoms include moderate to severe pain usually on one side of the head, pain that throbs or pulses, sensitivity to light and sound, nausea and vomiting, and blurred vision or lightheadedness.",
            "triggers": "Common migraine triggers include hormonal changes, certain foods and drinks, stress, sensory stimuli, sleep changes, physical exertion, weather changes, and certain medications.",
            "treatment": "Treatment aims to relieve symptoms and prevent future attacks. Options include pain-relieving medications, preventive medications, lifestyle changes, and alternative therapies."
        },
        "url": "https://www.mayoclinic.org/diseases-conditions/migraine"
    },
    {
        "source": "Medical Knowledge Base",
        "title": "Hypertension (High Blood Pressure)",
        "category": "Cardiovascular Disorders",
        "sections": {
            "overview": "High blood pressure is a common condition that affects the body's arteries. With high blood pressure, the force of blood pushing against artery walls is consistently too high. Normal blood pressure is below 120/80 mm Hg.",
            "symptoms": "Most people with high blood pressure have no symptoms. In severe cases: headaches, shortness of breath, or nosebleeds may occur.",
            "risk_factors": "Risk factors include age, family history, being overweight or obese, lack of physical activity, tobacco use, too much salt, too little potassium, drinking too much alcohol, and stress.",
            "prevention": "Prevention strategies include eating a healthy diet with less salt, getting regular physical activity, maintaining a healthy weight, limiting alcohol, not smoking, and managing stress.",
            "treatment": "Treatment includes lifestyle changes and medications such as diuretics, ACE inhibitors, angiotensin II receptor blockers, and calcium channel blockers."
        },
        "url": "https://www.mayoclinic.org/diseases-conditions/high-blood-pressure"
    },
    {
        "source": "Medical Knowledge Base",
        "title": "Asthma",
        "category": "Respiratory Disorders",
        "sections": {
            "overview": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, wheezing, and shortness of breath.",
            "symptoms": "Common symptoms include shortness of breath, chest tightness or pain, wheezing when exhaling, trouble sleeping caused by shortness of breath, and coughing or wheezing attacks.",
            "triggers": "Common triggers include airborne allergens, respiratory infections, physical activity, cold air, air pollutants, certain medications, strong emotions, and stress.",
            "treatment": "Treatment includes long-term control medications such as inhaled corticosteroids, and quick-relief medications like short-acting beta agonists."
        },
        "url": "https://www.mayoclinic.org/diseases-conditions/asthma"
    },
    {
        "source": "Medical Knowledge Base",
        "title": "Anxiety Disorders",
        "category": "Mental Health",
        "sections": {
            "overview": "People with anxiety disorders frequently have intense, excessive, and persistent worry and fear about everyday situations. These feelings interfere with daily activities and can last a long time.",
            "symptoms": "Common symptoms include feeling nervous or tense, having a sense of impending danger, increased heart rate, rapid breathing, sweating, trembling, feeling weak or tired, and trouble concentrating.",
            "types": "Types include generalized anxiety disorder, panic disorder, social anxiety disorder, and specific phobias.",
            "treatment": "Main treatments are psychotherapy (cognitive behavioral therapy), medications (antidepressants, anti-anxiety medications), and lifestyle changes."
        },
        "url": "https://www.mayoclinic.org/diseases-conditions/anxiety"
    }
]

# Save sample data
data_file = Path("data/raw/medical_knowledge_base.json")
with open(data_file, 'w', encoding='utf-8') as f:
    json.dump(sample_medical_data, f, indent=2, ensure_ascii=False)

logger.info(f"✅ Created {len(sample_medical_data)} sample documents")

# Step 2: Process into chunks
print("\n⚙️  Step 2: Processing documents into chunks...")

all_chunks = []
chunk_id = 0

for doc in sample_medical_data:
    title = doc['title']
    
    for section_name, section_content in doc['sections'].items():
        # Split into sentences
        sentences = section_content.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|').split('|')
        
        # Group into ~500 char chunks
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > 500 and current_chunk:
                chunk_text = f"{title} - {section_name.title()}\n\n{' '.join(current_chunk)}"
                
                all_chunks.append({
                    'chunk_id': f"{doc['title'].lower().replace(' ', '_')}_chunk_{chunk_id}",
                    'text': chunk_text,
                    'metadata': {
                        'document_title': title,
                        'section': section_name,
                        'source': doc['source'],
                        'category': doc['category'],
                        'url': doc.get('url', ''),
                        'chunk_index': chunk_id
                    }
                })
                
                chunk_id += 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Add remaining
        if current_chunk:
            chunk_text = f"{title} - {section_name.title()}\n\n{' '.join(current_chunk)}"
            
            all_chunks.append({
                'chunk_id': f"{doc['title'].lower().replace(' ', '_')}_chunk_{chunk_id}",
                'text': chunk_text,
                'metadata': {
                    'document_title': title,
                    'section': section_name,
                    'source': doc['source'],
                    'category': doc['category'],
                    'url': doc.get('url', ''),
                    'chunk_index': chunk_id
                }
            })
            
            chunk_id += 1

# Save chunks
chunks_file = Path("data/processed/all_chunks.json")
with open(chunks_file, 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

logger.info(f"✅ Created {len(all_chunks)} chunks")

# Step 3: Build ChromaDB - WITHOUT OpenAI embedding function
print("\n🗄️  Step 3: Building ChromaDB...")

try:
    import chromadb
    from chromadb.config import Settings
    
    # Initialize with default embedding function (sentence-transformers)
    chroma_dir = Path("data/vector_db/chroma")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Delete if exists
    try:
        client.delete_collection(name="medical_knowledge")
        logger.info("Deleted existing collection")
    except:
        pass
    
    # Create collection - let ChromaDB use default embedding
    collection = client.create_collection(
        name="medical_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add in batches
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        
        documents = [chunk['text'] for chunk in batch]
        metadatas = [chunk['metadata'] for chunk in batch]
        ids = [chunk['chunk_id'] for chunk in batch]
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added batch {i//batch_size + 1}")
    
    count = collection.count()
    logger.info(f"✅ ChromaDB created with {count} documents")
    
    # Test query
    print("\n🔍 Testing query...")
    results = collection.query(
        query_texts=["What are the symptoms of diabetes?"],
        n_results=3
    )
    
    logger.info(f"✅ Query successful! Retrieved {len(results['documents'][0])} docs")
    
except Exception as e:
    logger.error(f"❌ ChromaDB setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ SETUP COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • Documents: {len(sample_medical_data)}")
print(f"   • Chunks: {len(all_chunks)}")
print(f"   • Database: {count} indexed")
print("\n🚀 Run: streamlit run src/ui/app_simple.py")
print("="*70)