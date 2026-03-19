import streamlit as st
import requests
import json
import os

# Backend URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="iCog Labs RAG", layout="wide")

# --- UI Header ---
st.markdown("""
    <div style="background-color:#0E1117; padding:20px; border-radius:10px; border-bottom: 4px solid #00D1FF;">
        <h2 style="color:white;">🚀 iCog Labs: Professional RAG System</h2>
        <p style="color:#00D1FF;">Advanced Semantic Chunking | Multi-Query RRF | Streaming API</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar for Ingestion ---
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_files = st.file_uploader("Upload Knowledge PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Build Knowledge Base"):
        if not uploaded_files:
            st.error("Please upload PDFs first.")
        else:
            with st.spinner("Processing & Embedding..."):
                files = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
                try:
                    response = requests.post(f"{API_URL}/initialize", files=files)
                    if response.status_code == 200:
                        st.success("Vector Store Ready!")
                        st.session_state.ready = True
                    else:
                        st.error("Initialization Failed.")
                except Exception as e:
                    st.error(f"Backend unreachable: {e}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Call FastAPI with streaming enabled
        try:
            with requests.post(
                f"{API_URL}/ask", 
                json={"question": prompt, "history": []}, 
                stream=True
            ) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")