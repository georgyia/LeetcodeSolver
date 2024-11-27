from classes.problem import Problem
from tools.llama_templates import make_instruct_prompt
from tools.helpers import make_llm_request
from aleph_alpha_client import Client

class ShitFilter:
    def __init__(self, problem: Problem, client: Client, model: str):
        self.problem = problem
        self.client = client
        self.model = model

    def create_prompt(self):
        system_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Tools: None

You are a world-class LeetCode expert and a logical problem reformulation assistant. Your task is to process problem statements that may include irrelevant or extraneous information and provide an equivalent but streamlined and structured formulation of the problem. Follow these guidelines:

1. **Filter Unnecessary Information**: Remove all extraneous details that are not directly related to solving the problem.
2. **Formal and Structured Reformulation**: Reformulate the problem into a clear, concise, and logically equivalent statement. Use mathematical notations or precise programming terms when applicable.
3. **Exact Equivalence**: Ensure the new formulation is mathematically and logically identical to the original problem.
4. **Output Format**: Always use plain text to describe the reformulated problem.

For every reformulated problem, provide the following structure:
- **Original Problem**: Include a brief summary or key components of the original input for reference.
- **Reformulated Problem**: Present the structured and equivalent version here.

You must ensure clarity, precision, and logical rigor in your responses. Begin your task now.<|eot_id|>
"""

        return make_instruct_prompt(system_prompt, self.problem.question)

    def filter(self) -> str:
        prompt = self.create_prompt()
        print(f"Filtering Shit")
        return make_llm_request(prompt, self.client, self.model)
