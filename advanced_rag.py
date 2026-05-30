import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from datetime import datetime
import os
import hashlib

st.set_page_config(page_title="Advanced RAG System", page_icon="🚀", layout="wide")

st.title("🚀 Advanced RAG System")
st.markdown("*Hybrid Search + Evaluation + Feedback + Caching*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(
        name="knowledge_base",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
    )
    return collection, client

@st.cache_resource
def load_bm25():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("knowledge_base")
    all_docs = collection.get()
    
    tokenized_docs = [doc.lower().split() for doc in all_docs['documents']]
    bm25 = BM25Okapi(tokenized_docs)
    
    return bm25, all_docs['documents'], all_docs['metadatas']

def hybrid_search(query, collection, bm25, all_docs, all_metadatas, top_k=3, alpha=0.5):
    cache_key = hashlib.md5(f"{query}_{alpha}".encode()).hexdigest()
    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]
    
    vector_results = collection.query(query_texts=[query], n_results=top_k*2)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    combined_results = []
    for i in range(len(vector_results['documents'][0])):
        vector_score = 1 - vector_results['distances'][0][i]
        
        doc_text = vector_results['documents'][0][i]
        try:
            doc_idx = all_docs.index(doc_text)
            bm25_score = bm25_scores[doc_idx] / (max(bm25_scores) + 0.001)
        except:
            bm25_score = 0
        
        combined_score = alpha * vector_score + (1 - alpha) * bm25_score
        
        combined_results.append({
            'document': doc_text,
            'metadata': vector_results['metadatas'][0][i],
            'vector_score': vector_score,
            'bm25_score': bm25_score,
            'combined_score': combined_score
        })
    
    combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
    st.session_state.cache[cache_key] = combined_results[:top_k]
    
    return combined_results[:top_k]

try:
    collection, client = load_db()
    bm25, all_docs, all_metadatas = load_bm25()
    st.success("✅ Advanced RAG system ready! (Hybrid search + BM25)")
except Exception as e:
    st.error(f"Error loading: {e}")
    st.info("Run 'python rag_system_simple.py' first")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Advanced Settings")
    alpha = st.slider("Hybrid Search Weight", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    st.markdown("## 📊 Evaluation Dashboard")
    
    if st.button("Show Evaluation Metrics"):
        if st.session_state.evaluation_results:
            avg_relevancy = np.mean([e.get('answer_relevancy', 0) for e in st.session_state.evaluation_results])
            st.metric("Avg Answer Relevancy", f"{avg_relevancy:.1%}")
            st.metric("Total Evaluations", len(st.session_state.evaluation_results))
        else:
            st.info("Ask questions to see metrics")
    
    st.markdown("---")
    st.markdown("## 👍 User Feedback Summary")
    
    if st.session_state.feedback:
        helpful_count = sum(1 for f in st.session_state.feedback if f.get('helpful', False))
        satisfaction = helpful_count / len(st.session_state.feedback)
        st.metric("User Satisfaction", f"{satisfaction:.1%}")
        st.metric("Total Feedback", len(st.session_state.feedback))
    
    st.markdown("---")
    if st.button("🗑️ Clear Cache"):
        st.session_state.cache = {}
        st.rerun()

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Question input
question = st.chat_input("Ask your question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = hybrid_search(question, collection, bm25, all_docs, all_metadatas, alpha=alpha)
            
            if results:
                best = results[0]
                answer = best['document']
                source = best['metadata']['source']
                confidence = best['combined_score'] * 100
                
                if confidence < 30:
                    answer = "⚠️ Low confidence. Please rephrase."
                    source = "N/A"
                
                st.markdown(answer)
                st.caption(f"📚 Source: {source} | 🎯 Confidence: {confidence:.1f}%")
                
                with st.expander("🔍 See retrieval details"):
                    st.write(f"**Semantic score:** {best['vector_score']:.2f}")
                    st.write(f"**Keyword score (BM25):** {best['bm25_score']:.2f}")
                    st.write(f"**Combined score:** {best['combined_score']:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"help_{len(st.session_state.messages)}"):
                        st.session_state.feedback.append({
                            "question": question, "confidence": confidence, "helpful": True
                        })
                        st.success("Thanks!")
                with col2:
                    if st.button("👎 Not helpful", key=f"nothelp_{len(st.session_state.messages)}"):
                        st.session_state.feedback.append({
                            "question": question, "confidence": confidence, "helpful": False
                        })
                        st.warning("Feedback recorded")
                
                st.session_state.evaluation_results.append({
                    "answer_relevancy": best['combined_score']
                })
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{answer}\n\n*Source: {source} | Confidence: {confidence:.1f}%*"
                })
            else:
                st.error("No results found")