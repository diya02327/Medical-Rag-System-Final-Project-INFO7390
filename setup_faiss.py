"""
Medical RAG Setup - FAISS Only (No ChromaDB, No Proxy Issues)
"""
import json
import numpy as np
from pathlib import Path
import pickle

print("="*70)
print("MEDICAL RAG SETUP - FAISS ONLY")
print("="*70)

# Create directories
dirs = ["data/raw", "data/processed", "data/vector_db"]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {d}")

# Medical data
print("\n📦 Creating medical knowledge base...")
medical_data = [
    {
        "title": "Type 2 Diabetes",
        "sections": {
            "overview": "Type 2 diabetes is a chronic condition affecting how your body metabolizes sugar (glucose). Your body either resists the effects of insulin or doesn't produce enough insulin to maintain normal glucose levels.",
            "symptoms": "Increased thirst and frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, frequent infections, numbness or tingling in hands or feet, areas of darkened skin.",
            "causes": "Type 2 diabetes develops when the body becomes resistant to insulin or when the pancreas is unable to produce enough insulin. Risk factors include weight, inactivity, family history, age, and gestational diabetes.",
            "treatment": "Treatment focuses on managing blood sugar levels through diet, exercise, and medication. Some people may require insulin therapy. Regular monitoring and lifestyle changes are essential.",
            "prevention": "Eat healthy foods, get active, lose excess weight, and avoid sedentary behaviors. Regular check-ups help detect prediabetes before it progresses to type 2 diabetes."
        }
    },
    {
        "title": "Migraine Headaches",
        "sections": {
            "overview": "A migraine is a headache that causes severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
            "symptoms": "Pain on one side of your head, throbbing or pulsing pain, sensitivity to light and sounds, nausea and vomiting, blurred vision, lightheadedness, sometimes followed by fainting.",
            "triggers": "Hormonal changes in women, certain foods and drinks (aged cheeses, salty foods, alcohol), stress, sensory stimuli (bright lights, loud sounds), sleep changes, physical factors, weather changes.",
            "treatment": "Pain-relieving medications taken during migraine attacks, preventive medications taken regularly, lifestyle changes to avoid triggers, alternative therapies like acupuncture and biofeedback.",
            "prevention": "Identify and avoid triggers, maintain regular sleep schedule, manage stress, stay hydrated, exercise regularly, avoid skipping meals."
        }
    },
    {
        "title": "Hypertension (High Blood Pressure)",
        "sections": {
            "overview": "High blood pressure is a common condition where the force of blood against artery walls is consistently too high. It can damage blood vessels and organs over time if left untreated.",
            "symptoms": "Most people have no symptoms even with dangerously high blood pressure. Some may experience headaches, shortness of breath, or nosebleeds, but these aren't specific and usually occur when blood pressure reaches dangerous levels.",
            "risk_factors": "Age, race, family history, being overweight or obese, lack of physical activity, tobacco use, too much salt, too little potassium, excessive alcohol consumption, stress, chronic conditions.",
            "complications": "Heart attack or stroke, aneurysm, heart failure, weakened blood vessels in kidneys, thickened or torn blood vessels in eyes, metabolic syndrome, memory problems.",
            "prevention": "Eat a healthy diet with less salt, exercise regularly, maintain a healthy weight, limit alcohol, don't smoke, manage stress, monitor blood pressure at home, get regular checkups.",
            "treatment": "Lifestyle changes combined with medications such as diuretics, ACE inhibitors, angiotensin II receptor blockers, calcium channel blockers, or beta blockers depending on individual needs."
        }
    },
    {
        "title": "Asthma",
        "sections": {
            "overview": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, wheezing, and shortness of breath.",
            "symptoms": "Shortness of breath, chest tightness or pain, wheezing when exhaling, trouble sleeping caused by shortness of breath, coughing or wheezing attacks worsened by respiratory viruses like cold or flu.",
            "triggers": "Airborne allergens (pollen, dust mites, mold, pet dander), respiratory infections, physical activity, cold air, air pollutants and irritants like smoke, certain medications, strong emotions and stress.",
            "diagnosis": "Physical exam, lung function tests (spirometry and peak flow), imaging tests (chest X-ray), allergy testing, blood tests to check eosinophil levels.",
            "treatment": "Long-term control medications (inhaled corticosteroids, leukotriene modifiers, combination inhalers), quick-relief medications (short-acting beta agonists), allergy medications if needed."
        }
    },
    {
        "title": "Anxiety Disorders",
        "sections": {
            "overview": "Anxiety disorders involve more than temporary worry or fear. For people with anxiety disorders, the anxiety doesn't go away and can get worse over time, interfering with daily activities.",
            "symptoms": "Feeling nervous, restless or tense, having a sense of impending danger, increased heart rate, rapid breathing, sweating, trembling, feeling weak or tired, trouble concentrating, difficulty sleeping.",
            "types": "Generalized anxiety disorder (GAD), panic disorder, social anxiety disorder, specific phobias, separation anxiety disorder, agoraphobia.",
            "causes": "Medical conditions (heart disease, diabetes, thyroid problems), genetics, brain chemistry, environmental stressors, trauma.",
            "treatment": "Psychotherapy (cognitive behavioral therapy is most effective), medications (antidepressants, buspirone, benzodiazepines for short-term relief), lifestyle changes including stress management and relaxation techniques.",
            "prevention": "Get help early if you notice symptoms, stay physically active, avoid alcohol and recreational drugs, manage stress through meditation or yoga, prioritize sleep."
        }
    }
]

# Save raw data
with open("data/raw/medical_knowledge.json", 'w') as f:
    json.dump(medical_data, f, indent=2)
print(f"✅ Saved {len(medical_data)} medical documents")

# Create chunks
print("\n⚙️  Creating text chunks...")
chunks = []
chunk_id = 0

for doc in medical_data:
    for section_name, content in doc['sections'].items():
        chunk = {
            'id': f"chunk_{chunk_id}",
            'text': f"{doc['title']} - {section_name.title()}\n\n{content}",
            'metadata': {
                'document_id': doc['title'].lower().replace(' ', '_'),
                'title': doc['title'],
                'section': section_name,
                'source': 'Medical Knowledge Base'
            }
        }
        chunks.append(chunk)
        chunk_id += 1

with open("data/processed/chunks.json", 'w') as f:
    json.dump(chunks, f, indent=2)
print(f"✅ Created {len(chunks)} text chunks")

# Generate embeddings
print("\n📊 Generating embeddings with SentenceTransformer...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk['text'] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

print(f"✅ Generated embeddings: {embeddings.shape}")

# Build FAISS index
print("\n🗄️  Building FAISS vector index...")
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings.astype('float32'))

print(f"✅ FAISS index built: {index.ntotal} vectors, dimension {dimension}")

# Save everything
print("\n💾 Saving to disk...")
faiss.write_index(index, "data/vector_db/faiss_index.bin")
print("✅ Saved FAISS index")

with open("data/vector_db/chunks.pkl", 'wb') as f:
    pickle.dump(chunks, f)
print("✅ Saved chunks")

# Test query
print("\n🔍 Testing search functionality...")
test_query = "What are the symptoms of diabetes?"
test_embedding = model.encode([test_query]).astype('float32')
distances, indices = index.search(test_embedding, 3)

print(f"\n✅ Search test successful! Top 3 results for: '{test_query}'")
for i, idx in enumerate(indices[0][:3]):
    chunk = chunks[idx]
    print(f"   {i+1}. {chunk['metadata']['title']} - {chunk['metadata']['section']}")
    print(f"      Distance: {distances[0][i]:.4f}")

print("\n" + "="*70)
print("✅ SETUP COMPLETE!")
print("="*70)
print("\n📊 Summary:")
print(f"   • Medical documents: {len(medical_data)}")
print(f"   • Text chunks: {len(chunks)}")
print(f"   • Vector dimension: {dimension}")
print(f"   • FAISS index: {index.ntotal} vectors")
print("\n🚀 Next: Run the app with: streamlit run app_medical.py")
print("="*70)