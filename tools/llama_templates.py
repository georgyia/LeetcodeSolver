
def make_instruct_prompt(system_prompt: str, user_prompt: str) -> str:
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#user-and-assistant-conversation
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    return prompt


def make_tool_calling_prompt(system_prompt: str, user_prompt: str, tools: list[dict]) -> str:
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#builtin-tool-calling
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Environment: ipython
    Tools: {tools}
    Cutting Knowledge Date: December 2023
    Today Date: 21 September 2024

    {system_prompt}
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    return prompt

def make_code_interpreter_prompt(system_prompt: str, user_prompt: str) -> str:
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#builtin-code-interpreter
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Environment: ipython
    {system_prompt}
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    return prompt


def original_prompt_template(problem: dict) -> str:
    """
    Generate a prompt for a given problem.

    Args:
        problem (dict): A dictionary containing the problem, test cases, and starter code.

    Returns:
        str: A prompt for the given problem.
    """

    prompt = "\nQUESTION:\n"
    prompt += problem["question"]
    prompt += "\n\nSTARTER CODE:\n"
    prompt += problem["starter_code"]
    prompt += "\n\nWrite only the code, directly executable, no tests or other comments."
    prompt += " Don't use markdown code blocks."
    prompt += "\n\nCODE:\n"
    return prompt