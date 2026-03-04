import streamlit as st
import os

# Set API key from Streamlit secrets if running in cloud, else local env
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.rag.retriever import VectorStore
from src.rag.generator import generate_answer

# Page config
st.set_page_config(
    page_title="Ops Copilot",
    page_icon="🏭",
    layout="centered"
)

# Header
st.title("🏭 Ops Copilot")
st.markdown("Ask questions about manufacturing operations in plain English. Answers are grounded in real Fanuc robot SOPs.")

# --- Initialize Backend in Streamlit ---
@st.cache_resource
def load_backend():
    db = VectorStore()
    db.load_documents_from_folder("data")
    return db

with st.spinner("Loading manufacturing SOPs and Fanuc manuals..."):
    db = load_backend()

# Initialize chat history and conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []  # stores {"question": ..., "answer": ...}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Source Citations"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** (relevance: {src['score']:.4f})")
                    st.caption(src["text"])

# Quick example questions in sidebar
with st.sidebar:
    st.header("💡 Try asking:")
    st.markdown("- *What vibration level requires immediate machine shutdown?*")
    st.markdown("- *What is the maximum torque for spindle bolts?*")
    st.markdown("- *How do I perform a First Article Inspection?*")
    st.markdown("- *What are the E-stop recovery steps?*")
    st.markdown("- *What is the motion range of axis J1?*")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Your question is searched against 500+ pages of Fanuc robot manuals")
    st.markdown("2. Hybrid retrieval: semantic (ChromaDB) + keyword (BM25)")
    st.markdown("3. Results merged via Reciprocal Rank Fusion")
    st.markdown("4. GPT-4o-mini generates a grounded answer")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about machine operations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching SOPs...")
        
        try:
            # 1. Direct Hybrid Retrieval
            results = db.search(query=prompt, n_results=3) 
            chunks = [r["text"] for r in results]
            
            # 2. Direct OpenAI Generation
            answer = generate_answer(prompt, chunks, history=st.session_state.memory)
            
            # 3. Format sources
            sources = [
                {"text": r["text"][:200] + "...", "source": r["source"], "score": r["score"]}
                for r in results
            ]
            
            message_placeholder.markdown(answer)
            
            if sources:
                with st.expander("📄 Source Citations"):
                    for src in sources:
                        st.markdown(f"**{src['source']}** (relevance: {src['score']:.4f})")
                        st.caption(src["text"])
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
            st.session_state.memory.append({"question": prompt, "answer": answer})
            if len(st.session_state.memory) > 5:
                st.session_state.memory = st.session_state.memory[-5:]
                
        except Exception as e:
            message_placeholder.markdown(f"⚠️ **Error:** {str(e)}")
