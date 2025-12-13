import os

# Define all directories and files
project_structure = {
    "medical-rag-assistant/": [
        "requirements.txt",
        ".env.example",
        "README.md",
        "run.py",
    ],

    "medical-rag-assistant/data/": [],
    "medical-rag-assistant/data/raw/": [],
    "medical-rag-assistant/data/processed/": [],
    "medical-rag-assistant/data/vector_db/": [],

    "medical-rag-assistant/src/": ["__init__.py"],
    
    "medical-rag-assistant/src/data_collection/": [
        "__init__.py",
        "medical_sources.py",
        "scraper.py"
    ],
    
    "medical-rag-assistant/src/preprocessing/": [
        "__init__.py",
        "document_processor.py",
        "chunking.py"
    ],
    
    "medical-rag-assistant/src/vector_db/": [
        "__init__.py",
        "chroma_manager.py",
        "faiss_manager.py",
        "embeddings.py"
    ],
    
    "medical-rag-assistant/src/llm/": [
        "__init__.py",
        "rag_pipeline.py",
        "prompts.py"
    ],

    "medical-rag-assistant/src/ui/": [
        "__init__.py",
        "app.py"
    ],

    "medical-rag-assistant/tests/": ["__init__.py"],
    "medical-rag-assistant/tests/": ["__init__.py"],
    "medical-rag-assistant/tests/": [],
    "medical-rag-assistant/tests/": ["__init__.py", "test_retrieval.py", "test_generation.py"],

    "medical-rag-assistant/notebooks/": [
        "01_data_exploration.ipynb",
        "02_vector_db_setup.ipynb",
        "03_evaluation.ipynb"
    ],

    "medical-rag-assistant/config/": [
        "config.yaml"
    ]
}


def create_project(structure):
    for directory, files in structure.items():
        # Create directories
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

        # Create files
        for file in files:
            file_path = os.path.join(directory, file)
            with open(file_path, "w") as f:
                # Add starter content
                if file.endswith("__init__.py"):
                    f.write("# Init file\n")
                elif file == "README.md":
                    f.write("# Medical RAG Assistant\n\nGenerated project structure.\n")
                elif file == ".env.example":
                    f.write("OPENAI_API_KEY=\n")
                elif file == "requirements.txt":
                    f.write(
                        "openai\nchromadb\nfaiss-cpu\nstreamlit\npydantic\npython-dotenv\n"
                    )
                elif file == "config.yaml":
                    f.write("embedding_model: text-embedding-3-small\n")
                else:
                    f.write(f"# Placeholder for {file}\n")

            print(f"  Created file: {file_path}")


# Run the generator
if __name__ == "__main__":
    create_project(project_structure)
    print("\n🎉 Project structure generated successfully!")
