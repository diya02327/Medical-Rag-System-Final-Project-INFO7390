"""
Debug script to check app components
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

print("=" * 50)
print("MEDICAL RAG ASSISTANT - DEBUG")
print("=" * 50)

# Load environment
load_dotenv()
print("\n1. Environment Check:")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"   ✅ OpenAI API Key found: {api_key[:10]}...")
else:
    print("   ❌ OpenAI API Key NOT found!")
    print("   → Add OPENAI_API_KEY to .env file")

# Check data directories
print("\n2. Directory Check:")
dirs_to_check = [
    "data/raw",
    "data/processed", 
    "data/vector_db/chroma"
]

for dir_path in dirs_to_check:
    if Path(dir_path).exists():
        files = list(Path(dir_path).glob("*"))
        print(f"   ✅ {dir_path}: {len(files)} files")
    else:
        print(f"   ❌ {dir_path}: NOT FOUND")

# Check if sample data exists
print("\n3. Sample Data Check:")
sample_data_file = Path("data/raw/medical_knowledge_base.json")
if sample_data_file.exists():
    import json
    with open(sample_data_file) as f:
        data = json.load(f)
    print(f"   ✅ Sample data exists: {len(data)} documents")
else:
    print("   ❌ Sample data NOT found")
    print("   → Run: python run.py --setup")

# Check ChromaDB
print("\n4. ChromaDB Check:")
try:
    sys.path.append(str(Path(__file__).parent))
    from src.vector_db.chroma_manager import ChromaDBManager
    
    chroma = ChromaDBManager()
    chroma.create_collection()
    stats = chroma.get_collection_stats()
    
    print(f"   ✅ ChromaDB initialized")
    print(f"   📊 Documents indexed: {stats['document_count']}")
    
    if stats['document_count'] == 0:
        print("   ⚠️  WARNING: No documents in database!")
        print("   → Run: python run.py --setup")
    
except Exception as e:
    print(f"   ❌ ChromaDB Error: {e}")

# Test query
print("\n5. Test Query:")
if stats['document_count'] > 0:
    try:
        result = chroma.query("What is diabetes?", n_results=3)
        print(f"   ✅ Query successful")
        print(f"   📄 Retrieved {len(result['documents'])} documents")
    except Exception as e:
        print(f"   ❌ Query Error: {e}")
else:
    print("   ⏭️  Skipping (no documents)")

print("\n" + "=" * 50)
print("DIAGNOSIS COMPLETE")
print("=" * 50)

# Recommendations
print("\n📋 Recommendations:")
if not api_key:
    print("1. Create .env file and add: OPENAI_API_KEY=sk-your-key")
if not sample_data_file.exists() or stats.get('document_count', 0) == 0:
    print("2. Run: python run.py --setup")
print("3. Then run: streamlit run src/ui/app.py")