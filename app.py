import streamlit as st
from llama_cpp import Llama
import os
from datetime import datetime
import pandas as pd
from huggingface_hub import hf_hub_download

# Initialize Llama model
@st.cache_resource
def load_model():
    # Download model directly from Hugging Face
    model_path = hf_hub_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GGML",
        filename="llama-2-7b-chat.ggmlv3.q4_0.bin"
    )
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4
    )

llm = load_model()

# Page config
st.set_page_config(page_title="AI Friend", page_icon="🤝")
st.title("AI Friend")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    system_greeting = "Hi there! How are you feeling today? 😊"
    st.session_state.messages.append({"role": "assistant", "content": system_greeting})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Share your thoughts..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare conversation history
    conversation_history = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in st.session_state.messages[:-1]
    ])

    # Generate response
    system_prompt = """You are a mature and sophisticated friend having a deeply engaging conversation. Key attributes:
    - Highly empathetic and emotionally intelligent
    - Give concise but meaningful responses (2-3 sentences)
    - Ask thoughtful follow-up questions
    - Remember and reference previous parts of the conversation
    - Keep the conversation flowing naturally

    Previous Conversation:
    {conversation_history}
    """

    response = llm.create_completion(
        f"{system_prompt}\nUser: {prompt}\nAssistant:",
        max_tokens=256,
        stop=["User:", "\n"],
        temperature=0.7
    )
    bot_response = response['choices'][0]['text'].strip()

    # Display bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# Save conversation button
def save_conversation():
    if st.session_state.messages:
        df = pd.DataFrame([
            {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in st.session_state.messages
        ])
        filename = f'conversations/chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        os.makedirs('conversations', exist_ok=True)
        df.to_csv(filename, index=False)

if st.sidebar.button("Save Conversation"):
    save_conversation()