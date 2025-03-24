import streamlit as st
from rag_chatbot import RAGChatbot
import time

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    return RAGChatbot()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("""
This is a proof of concept interface for the RAG Chatbot. Ask any question and get answers based on the indexed documents.
""")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response from chatbot
            chatbot = get_chatbot()
            response = chatbot.chat(prompt)
            
            # Display response
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a Streamlit interface for the RAG Chatbot that uses:
    - Multiple specialized agents
    - Local knowledge retrieval
    - Web search capabilities
    - Quality evaluation pipeline
    """)
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun() 