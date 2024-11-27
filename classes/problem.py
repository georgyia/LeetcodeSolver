from pydantic import BaseModel

class Problem(BaseModel):
    problem_id: int
    question: str
    input_output: dict
    url: str
    difficulty: str  # fixme: convert to enum
    starter_code: str

