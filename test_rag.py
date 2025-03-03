from query_data_with_hist import query_rag_with_history
from langchain_ollama import OllamaLLM

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false', and a reasoning) Does the actual response have the same meaning as the expected response?
"""
import os
os.environ["LANGSMITH_TRACING"] = "true"

def test_mas_knowledge():
    assert query_and_validate(
        question="What is MAS? (Just give the full name without explanation",
        expected_response="Multi Agent System",
    )

def test_agent_knowledge():
    assert query_and_validate(
        question="State the properties of agent (strong notion) without further explanation",
        expected_response='Mentalistics notions like Beliefs & Intentions. Other properties include Veracity, Benevolence, Rationality, Mobility',
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag_with_history(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    model = OllamaLLM(model="deepseek-r1:7b")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
if __name__ == "__main__":
    test_mas_knowledge()
    test_agent_knowledge()

