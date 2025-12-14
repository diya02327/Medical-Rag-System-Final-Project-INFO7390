# 🏥 Medical Information RAG Assistant  
### AI-Powered Medical Information System Using Retrieval-Augmented Generation

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [Evaluation & Testing](#-evaluation--testing)
- [Ethical Considerations](#-ethical-considerations)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Demo](#-demo)
- [Contributors](#-contributors)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Overview

The **Medical Information RAG Assistant** is an AI-powered system that provides **trustworthy, evidence-based medical information** using **Retrieval-Augmented Generation (RAG)**.

Unlike generic health websites that provide inconsistent or fear-based content, this system retrieves information **exclusively from reputable medical sources** and generates **clear, patient-friendly responses with citations**.

**Course:** INFO 7390 – Advanced Data Science and Architecture  
**Institution:** Northeastern University  
**Semester:** Fall 2025

---

## 🔍 Problem Statement

### The Challenge

When people search for health information online, they encounter major issues:

- **Inconsistent Information:** Conflicting advice across websites  
- **Unreliable Sources:** Lack of medical credibility  
- **Fear-Based Content:** Clickbait-driven misinformation  
- **No Citations:** Poor traceability to original research  
- **Outdated Data:** Information without timestamps  

### The Impact

According to Pew Research Center:
- **72%** of internet users search for health information online  
- Only **15%** check the source and date  

This results in:
- Increased anxiety and fear  
- Incorrect self-diagnosis  
- Delayed or inappropriate care  
- Poor doctor–patient communication  

---

## 💡 Solution

This project builds an AI assistant that:

- Uses **only reputable medical sources**
- Provides **evidence-based information**
- Includes **clear citations**
- **Never diagnoses**
- Encourages consultation with healthcare professionals

### How It Works

The system uses **Retrieval-Augmented Generation (RAG)**:

- **Semantic Search:** FAISS retrieves relevant medical content  
- **LLM Generation:** GPT-4 produces clear, grounded responses  
- **Safety Guardrails:** Prompt engineering enforces medical safety  

---

## ✨ Key Features

### Core Functionality
- 🔍 Semantic search (meaning-based)
- 📚 Source citations on every answer
- 🏥 Safety-first medical design
- ⚡ Sub-second retrieval
- 💬 Patient-friendly language

### User Experience
- Interactive chat interface
- Query classification (symptoms vs conditions)
- Source transparency
- Emergency detection
- Clear disclaimers

### Technical Features
- FAISS vector database
- Sentence Transformers embeddings
- GPT-4 integration
- Section-aware chunking
- Automated testing

---

## 🧠 Architecture

**Pipeline Flow:**

User Query
↓
Embedding (384-dim vectors)
↓
FAISS Semantic Search (Top-K)
↓
Context Building with Sources
↓
GPT-4 with Safety Prompts
↓
Cited Response + Disclaimer

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|---------|-----------|---------|--------|
| Vector Search | FAISS | 1.9.0 | Fast similarity search |
| Embeddings | Sentence Transformers | 2.3.0 | Semantic encoding |
| LLM | OpenAI GPT-4 | API 1.12.0 | Response generation |
| UI | Streamlit | 1.30.0 | Web interface |
| Language | Python | 3.13 | Core implementation |

### Supporting Libraries
- NumPy
- Pandas
- python-dotenv
- Pytest

---

## 🚀 Installation

### Prerequisites
- Python 3.13 (or 3.11+)
- OpenAI API Key
- 4GB+ RAM
- Internet connection

### Step 1: Clone Repository
```bash
git clone https://github.com/diya02327/Medical-Rag-System-Final-Project-INFO7390.git
cd Medical-Rag-System-Final-Project-INFO7390


---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|---------|-----------|---------|--------|
| Vector Search | FAISS | 1.9.0 | Fast similarity search |
| Embeddings | Sentence Transformers | 2.3.0 | Semantic encoding |
| LLM | OpenAI GPT-4 | API 1.12.0 | Response generation |
| UI | Streamlit | 1.30.0 | Web interface |
| Language | Python | 3.13 | Core implementation |

### Supporting Libraries
- NumPy
- Pandas
- python-dotenv
- Pytest

---

## 🚀 Installation

### Prerequisites
- Python 3.13 (or 3.11+)
- OpenAI API Key
- 4GB+ RAM
- Internet connection

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/medical-rag-assistant.git
cd medical-rag-assistant

### Step 2: Create Virtual Environment

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

### Step 3: Install Dependencies
pip install --upgrade pip
pip install httpx==0.27.0
pip install openai==1.12.0
pip install sentence-transformers==2.3.0
pip install faiss-cpu==1.9.0
pip install streamlit==1.30.0
pip install python-dotenv==1.0.0

### Step 4: Configure Environment
cp .env.example .env
# Add OPENAI_API_KEY in .env

### Step 5: Build Knowledge Base
python setup_faiss.py

###Step 6: Run Application
streamlit run app_medical.py

---

📁 Project Structure

medical-rag-assistant/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_db/
│
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── vector_db/
│   ├── llm/
│   └── ui/
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_end_to_end.py
│
├── setup_faiss.py
├── app_medical.py
├── requirements_py313.txt
├── .env.example
├── README.md
└── LICENSE

---


# 📊 Data Sources

| Source        | Organization       | Credibility |
|---------------|--------------------|-------------|
| MedlinePlus   | NIH                | ⭐⭐⭐⭐⭐ |
| Mayo Clinic   | Mayo Foundation    | ⭐⭐⭐⭐⭐ |
| CDC           | Centers for Disease Control and Prevention (CDC) | ⭐⭐⭐⭐⭐ |

---

## 🩺 Covered Conditions

- Type 2 Diabetes  
- Migraines  
- Hypertension  
- Asthma  
- Anxiety Disorders  

---

# 🧪 Evaluation & Testing

## 🔍 Retrieval Metrics

- **Precision:** 85%  
- **Speed:** < 500 ms  
- **Coverage:** 82%  

## ✨ Generation Metrics

- **Citation Rate:** 95%  
- **Medical Disclaimer Rate:** 100%  
- **Unsafe Diagnosis Rate:** 0%  

---

## ▶️ Run Tests

```bash
python -m pytest tests/ -v
```

---

# ⚖️ Ethical Considerations

## ✅ What This System Does

- Provides **educational medical information**
- Uses and **cites reputable medical sources**
- Encourages users to **seek professional medical care**

## ❌ What This System Does NOT Do

- Diagnose medical conditions  
- Prescribe medication  
- Replace healthcare professionals  
- Provide emergency medical treatment  

---

## 🔐 Privacy & Transparency

- No personal data is stored  
- Queries are anonymous  
- Fully open-source and auditable  

---

# ⚠️ Limitations

- Covers only **five medical conditions**
- No real-time data updates  
- English-only support  
- Requires internet access  
- Cannot track individual medical history  

---

# 🚀 Future Improvements

- Expand the medical knowledge base  
- Add multilingual support  
- Enable real-time data updates  
- Introduce a voice-based interface  
- Improve emergency handling and guidance  

---

# 🎥 Demo

- 

---

# 👤 Contributors

- **Diya Gandhi** — Developer & Researcher  

---

# 🙏 Acknowledgments

- Mayo Clinic  
- Centers for Disease Control and Prevention (CDC)  
- National Institutes of Health (NIH)  
