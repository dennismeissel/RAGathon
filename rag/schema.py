# schema.py

from typing import List, Union, Literal, Optional
from pydantic import BaseModel, Field

class Question(BaseModel):
    text: str
    # The possible question kinds we expect:
    #   "number", "boolean", "name", "names"
    # If you also use "N/A" as a placeholder kind, add it below.
    kind: Literal["number", "boolean", "name", "names"]

class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(..., description="Physical page number in the PDF file")

class Answer(BaseModel):
    question_text: str = Field(..., description="Text of the question")
    kind: Literal["number", "name", "boolean", "names"] = Field(..., description="Kind of the question")
    # The answer can be float, string, bool, list of strings, or "N/A"
    value: Union[float, str, bool, List[str], Literal["N/A"]] = Field(..., description="Answer to the question")
    references: List[SourceReference] = Field([], description="References to the source material in the PDF file")

class AnswerSubmission(BaseModel):
    answers: List[Answer] = Field(..., description="List of answers to the questions")
    team_email: str = Field(..., description="Email that your team used to register for the challenge")
    submission_name: str = Field(..., description="Unique name of the submission (e.g. experiment name)")
    