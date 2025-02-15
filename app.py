import streamlit as st
import time
from query_data_with_hist import query_rag_with_history


# Streamed response emulator
def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        llm_answer = query_rag_with_history(prompt)
        response = st.write_stream(response_generator(llm_answer))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})