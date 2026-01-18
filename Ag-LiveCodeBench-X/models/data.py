from typing import TypedDict


# Type definitions
class Candidate(TypedDict):
    question_id: str
    solution: str
    reasoning: str


class Result(TypedDict):
    result: str
    question_id: str
    solution: str
    raw_exit_code: int
    raw_stdout: str
    raw_stderr: str


class RefinementTrainingExample(TypedDict):
    """Structure for training data from refinement process"""

    question_id: str
    language: str
    problem_statement: str
    original_code: str
    error_feedback: dict
    refined_code: str
    reasoning: str
