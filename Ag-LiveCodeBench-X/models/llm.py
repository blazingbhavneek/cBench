from pydantic import BaseModel, Field


# Pydantic models for structured outputs
class SolutionResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for solving the problem")
    solution: str = Field(description="The complete code solution in markdown format")


class RefinementResponse(BaseModel):
    reasoning: str = Field(
        description="Step-by-step analysis of the error and how to fix it"
    )
    refined_solution: str = Field(description="The corrected code in markdown format")
