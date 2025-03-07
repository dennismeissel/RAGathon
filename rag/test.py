# test.py

import rag

def main():
    # Example question and schema
    question = (
        ""
    )
    schema = "number"

    # Get the answer from our RAG workflow
    answer = rag.get_rag_answer(question, schema)

    # Print the final answer
    print("Final Answer:", answer)

if __name__ == "__main__":
    main()
