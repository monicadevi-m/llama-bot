import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datetime import datetime
import pandas as pd

# Initialize model
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much smaller, works on Streamlit Cloud
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

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

    # Format input
    input_text = f"{system_prompt}\nUser: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )
    
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = bot_response.split("Assistant:")[-1].strip()

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
