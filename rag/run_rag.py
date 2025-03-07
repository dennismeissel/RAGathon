# run_rag.py

import json
import logging

import rag
from schema import Question, AnswerSubmission, Answer, SourceReference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_duplicates(references: list[SourceReference]):
    """Removes duplicated references"""
    seen = set()
    new_objects = []
    for ref in references:
        identifier = ref.pdf_sha1 + str(ref.page_index)
        if identifier not in seen:
            seen.add(identifier)
            new_objects.append(ref)
    return new_objects

def main():
    # Read questions from questions.json
    with open("rag/questions.json", "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    # Parse them into a list of Question objects
    questions = [Question(**q) for q in raw_questions]
    total_questions = len(questions)  # Get total number of questions

    answers_list = []
    debug_result_list = []

    for idx, q in enumerate(questions, start=1):
        # Display progress
        logger.info(f"Processing question {idx}/{total_questions}...")

        # Safely call the RAG function
        try:
            rag_response = rag.get_rag_answer(q.text, q.kind)
            raw_answer = rag_response.get("answer", "N/A")
            llm_answer = rag_response.get("llm_answer", "llm_answer was not filled")
            raw_refs = rag_response.get("references", [])
        except Exception as e:
            logger.error(f"Connection error for question {idx}: {e}")
            # Mark this answer as ERROR and continue
            raw_answer = "ERROR"
            llm_answer = "ERROR"
            raw_refs = []

        # Convert references to our SourceReference model
        references = []
        for ref in raw_refs:
            pdf_sha1 = ref.get("filename", "")
            try:
                referenced_pages = str(ref.get("page", "")).split(",")
                page_indexes = [int(s) for s in referenced_pages]
            except ValueError:
                page_indexes = [0]
            for page_index in page_indexes:
                references.append(SourceReference(pdf_sha1=pdf_sha1, page_index=page_index))
        references = remove_duplicates(references)

        # Parse the answer according to its type
        if raw_answer == "ERROR":
            # If there's an error, just store "ERROR" and empty references
            value = "ERROR"
            references = []
        elif raw_answer == "N/A":
            value = "N/A"
            references = []
        else:
            if q.kind == "number":
                numeric_str = raw_answer.replace(",", "")
                try:
                    value = float(numeric_str)
                except ValueError:
                    value = "N/A"
            elif q.kind == "boolean":
                lower_answer = raw_answer.strip().lower()
                value = lower_answer in ["yes", "true"]
            elif q.kind == "names":
                value = [name.strip() for name in raw_answer.split(",") if name.strip()]
            elif q.kind == "name":
                value = raw_answer
            else:
                value = "N/A"

        # Create the answer object
        answer_obj = Answer(
            question_text=q.text,
            kind=q.kind,
            value=value,
            references=references
        )
        answers_list.append(answer_obj)

    print("\nAll questions processed!")

    # Build the submission object
    submission = AnswerSubmission(
        answers=answers_list
    )

    # Write results to results.json
    with open("rag/results.json", "w", encoding="utf-8") as f:
        f.write(submission.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
