import streamlit as st
import requests

# Set page config
st.set_page_config(
    page_title="Ops Copilot",
    page_icon="🤖",
    layout="centered"
)

# Header
st.title("🏭 Ops Copilot")
st.markdown("Ask questions in English. Get answers straight from the factory SOPs.")

# Backend API URL (FastAPI)
API_URL = "http://127.0.0.1:8000/ask"

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick example questions in sidebar
with st.sidebar:
    st.header("Try asking:")
    st.markdown("- *What vibration level requires immediate machine shutdown?*")
    st.markdown("- *What is the maximum torque for spindle bolts?*")
    st.markdown("- *How do I perform a First Article Inspection?*")

# Chat input
if prompt := st.chat_input("Ask a question about machine operations..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🔍")
        
        try:
            # Call the FastAPI backend
            payload = {"question": prompt}
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer returned.")
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error: Backend returned status code {response.status_code}"
                message_placeholder.markdown(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.markdown("⚠️ **Connection Error:** Ensure the FastAPI backend is running (`python -m uvicorn src.main:app`).")
