from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory


import argparse
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ["LANGSMITH_PROJECT"] = "ETS_monthly_example"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
System: You are a teacher that understands how intelligent agents works. 
You provide clear and detailed answers on material when you are asked by students, and elaborate on the concepts.
If there is an existing conversation, you may take it as context

History of this conversation:

{history}

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag_with_history(query_text)

def clean_response(response_text):
    start = response_text.find("<think>")
    end = response_text.find("</think>") + len("</think>")
    if start != -1 and end != -1:
        response_text = response_text[:start] + response_text[end:]
    return response_text.strip()


def query_rag_with_history(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = OllamaLLM(model="deepseek-r1:7b")
    # model = OllamaLLM(model="llama3.2:latest")
    
    chain = prompt_template | model

    chain_with_history = RunnableWithMessageHistory(
        chain,
        # Uses the get_by_session_id function defined in the example
        # above.
        get_by_session_id,
        input_messages_key="question",
        context_messages_key="context",
        history_messages_key="history",
    )
    response_text = chain_with_history.invoke(  # noqa: T201
        {"question": query_text, "context": context_text},
        config={"configurable": {"session_id": "foo"}}
    )

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    print(store)
    cleaned_response_text = clean_response(response_text)
    return cleaned_response_text

if __name__ == "__main__":
    main()

