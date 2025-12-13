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

