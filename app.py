import streamlit as st
import requests
from pathlib import Path

# Backend URL
API_URL = "http://127.0.0.1:8001"

# Supported file types for multimodal ingestion
ALLOWED_EXTENSIONS = ["pdf", "png", "jpg", "jpeg", "txt", "md", "csv"]

st.set_page_config(page_title="RAG System", layout="wide", initial_sidebar_state="expanded")

# --- Inject Custom CSS ---
def _load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()


def _get_mime_type(filename):
    ext = (filename or "").lower().split(".")[-1]
    mime = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "txt": "text/plain", "md": "text/markdown", "csv": "text/csv"}
    return mime.get(ext, "application/octet-stream")


# --- UI Header ---
st.markdown("""
<div class="rag-hero" style="
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    padding: 1.75rem 2rem;
    border-radius: 16px;
    border: 1px solid #334155;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
">
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
        <span style="font-size: 2rem;">🚀</span>
        <h1 style="color: #f1f5f9; margin: 0; font-size: 1.75rem; font-weight: 700;">RAG System</h1>
        <span style="
            background: linear-gradient(135deg, #00D1FF, #7C3AED);
            color: white;
            font-size: 0.7rem;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            font-weight: 600;
        ">LIVE</span>
    </div>
    <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">
        Semantic Chunking • Multi-Query RRF • History-Aware Chat • Multimodal (PDF, Images, Text)
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for Ingestion ---
with st.sidebar:
    st.markdown("### 📂 Document Ingestion")
    
    with st.expander("📄 Supported formats", expanded=False):
        st.caption("PDF, PNG, JPG, TXT, MD, CSV")
    
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, images, text)",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
    )
    
    if st.button("Build Knowledge Base", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload files first.")
        else:
            with st.spinner("Processing & embedding..."):
                file_tuples = []
                for f in uploaded_files:
                    f.seek(0)
                    file_tuples.append(("files", (f.name, f, _get_mime_type(f.name))))
                try:
                    response = requests.post(f"{API_URL}/initialize", files=file_tuples)
                    if response.status_code == 200:
                        data = response.json()
                        n = data.get("chunks", "?")
                        st.success(f"✓ Indexed **{n}** chunks")
                        st.session_state.ready = True
                    else:
                        try:
                            err = response.json()
                            d = err.get("detail", err.get("message", str(err)))
                            if isinstance(d, list):
                                msg = "; ".join(e.get("msg", str(e)) for e in d)
                            else:
                                msg = str(d)
                        except Exception:
                            msg = response.text or f"HTTP {response.status_code}"
                        st.error(f"**Initialization failed:** {msg}")
                except requests.exceptions.ConnectionError:
                    st.error(f"**Backend unreachable.** Is `uvicorn main:app --reload` running at {API_URL}?")
                except Exception as e:
                    st.error(f"**Error:** {e}")

    st.divider()
    st.markdown("### 💡 Features")
    # Export chat (unique feature)
    if st.session_state.get("messages"):
        chat_text = "\n\n".join(
            f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "📥 Export Chat",
            chat_text,
            file_name="rag_chat_export.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download conversation as Markdown",
        )
    
    st.caption(f"Backend: `{API_URL}`")
    
    with st.expander("ℹ️ About", expanded=False):
        st.caption("""
        **RAG System** uses:
        - Semantic chunking (topic-aware)
        - Multi-query expansion + RRF reranking
        - History-aware responses
        - PDF, images, text support
        """)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Top bar: status + New Chat
top_col1, top_col2, top_col3 = st.columns([1, 4, 1])
with top_col1:
    ready = st.session_state.get("ready", False)
    status_color = "#22c55e" if ready else "#64748b"
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="
            width: 8px; height: 8px; border-radius: 50%;
            background: {status_color};
            box-shadow: 0 0 8px {status_color};
        "></span>
        <span style="font-size: 0.85rem; color: #94a3b8;">{"Ready" if ready else "Upload docs to start"}</span>
    </div>
    """, unsafe_allow_html=True)
with top_col3:
    if st.button("🗑️ New Chat", help="Start a fresh conversation"):
        st.session_state.messages = []
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

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
        
        # Build history from previous messages (exclude current)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        
        try:
            with requests.post(
                f"{API_URL}/ask", 
                json={"question": prompt, "history": history}, 
                stream=True
            ) as r:
                if r.status_code != 200:
                    try:
                        err = r.json()
                        d = err.get("detail", err.get("error", str(err)))
                        msg = d if isinstance(d, str) else str(d)
                    except Exception:
                        msg = r.text or "Request failed"
                    st.error(msg)
                else:
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# Footer
st.markdown("<br>", unsafe_allow_html=True)
with st.container():
    st.markdown("""
    <div class="rag-footer">
        <strong>RAG System</strong> • Semantic Chunking • Multi-Query RRF • History-Aware • Multimodal
    </div>
    """, unsafe_allow_html=True)