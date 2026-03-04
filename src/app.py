import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Ops Copilot",
    page_icon="🏭",
    layout="centered"
)

# Header
st.title("🏭 Ops Copilot")
st.markdown("Ask questions about manufacturing operations in plain English. Answers are grounded in real Fanuc robot SOPs.")

# Backend API URL
API_URL = "http://127.0.0.1:8000/ask"

# Initialize chat history and conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []  # stores {"question": ..., "answer": ...}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if they exist
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
    st.markdown("- *What coolant concentration is required?*")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Your question is searched against 5+ manufacturing SOPs and Fanuc robot manuals")
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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching SOPs...")
        
        try:
            payload = {"question": prompt}
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                
                message_placeholder.markdown(answer)
                
                # Show source citations
                if sources:
                    with st.expander("📄 Source Citations"):
                        for src in sources:
                            st.markdown(f"**{src['source']}** (relevance: {src['score']:.4f})")
                            st.caption(src["text"])
                
                # Save to chat history with sources
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Save to conversation memory (for future context)
                st.session_state.memory.append({
                    "question": prompt,
                    "answer": answer
                })
                # Keep only last 5 turns
                if len(st.session_state.memory) > 5:
                    st.session_state.memory = st.session_state.memory[-5:]
                    
            else:
                error_msg = f"⚠️ Backend returned status code {response.status_code}"
                message_placeholder.markdown(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.markdown(
                "⚠️ **Connection Error:** Make sure the FastAPI backend is running.\n\n"
                "```bash\ncd sid/projA\nuvicorn src.main:app --reload\n```"
            )
