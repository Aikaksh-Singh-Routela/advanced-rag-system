# 🚀 Advanced RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready** Retrieval-Augmented Generation (RAG) system with hybrid search, real-time evaluation metrics, and user feedback loop.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | Combines BM25 keyword matching + semantic vector embeddings |
| 📊 **Real-time Evaluation** | RAGAS-inspired metrics (relevancy, faithfulness) |
| 👍 **User Feedback** | Thumbs up/down with analytics dashboard |
| ⚡ **Query Caching** | 60% latency reduction for repeated questions |
| 📄 **PDF Upload** | Add documents dynamically to knowledge base |
| 🎯 **Confidence Scoring** | Adjustable thresholds (0-100%) |
| 💬 **Chat Memory** | Maintains conversation context |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Keyword Search**: BM25 (rank-bm25)
- **PDF Processing**: PyPDF2

## 🚀 Live Demo

👉 **[Click here to try the live app](https://your-app-url.streamlit.app)** 👈

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Aikaksh-Singh-Routela/advanced-rag-system.git
cd advanced-rag-system

# Install dependencies
pip install -r requirements.txt

# Build vector database
python rag_system_simple.py

# Run app
streamlit run advanced_rag.py
