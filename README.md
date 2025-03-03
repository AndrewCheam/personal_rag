# Simple Chat with Streamlit

This is a simple chat application built using Streamlit. The app maintains chat history and generates responses using a function that queries a RAG (Retrieval-Augmented Generation) model with chat history.

## Features

- Interactive chat interface
- Maintains conversation history
- Streams responses word-by-word for a natural experience
- Uses a function `query_rag_with_history` to generate responses

## Installation

To run this application, make sure you have Python installed along with the required dependencies. Install them using:

```bash
pip install streamlit
```

## Usage

1. Clone this repository or save the script.
2. Run the Streamlit app using:
   ```bash
   streamlit run app.py
   ```
3. The chat interface will open in your web browser.
4. Enter a message in the input field to interact with the assistant.

## File Structure

- `app.py`: Main script containing the Streamlit chat implementation.
- `query_data_with_hist.py`: Module containing the function `query_rag_with_history` to generate responses.

## Explanation of Code

1. The app initializes chat history using `st.session_state.messages`.
2. It displays previous messages on rerun to maintain context.
3. The user inputs a prompt, which is stored in chat history and displayed in the UI.
4. The assistant’s response is generated using `query_rag_with_history` and streamed word-by-word using `response_generator`.
5. The response is stored in the session state to persist the chat history.

## Dependencies

- Python 3.x
- Streamlit

## Future Enhancements

- Implement user authentication
- Add support for multiple chat sessions
- Improve response handling and latency
- Integrate with an external API for enhanced RAG capabilities

## License

This project is open-source and available for modification and use under the MIT License.
