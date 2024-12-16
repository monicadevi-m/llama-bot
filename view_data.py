import streamlit as st
import pandas as pd
import os
import glob

def view_chat_history():
    st.title("Chat History Viewer")
    
    # Get all conversation files
    conversation_files = glob.glob('conversations/chat_*.csv')
    
    if conversation_files:
        # Sort files by date (newest first)
        conversation_files.sort(reverse=True)
        
        # Let user select a conversation
        selected_file = st.selectbox(
            'Select conversation to view:',
            conversation_files,
            format_func=lambda x: x.split('/')[-1]
        )
        
        if selected_file:
            # Load and display conversation
            df = pd.read_csv(selected_file)
            st.dataframe(df)
            
            # Add download button
            st.download_button(
                "Download this conversation",
                df.to_csv(index=False),
                f"download_{selected_file.split('/')[-1]}",
                "text/csv"
            )
    else:
        st.info("No saved conversations found")

if __name__ == "__main__":
    view_chat_history()