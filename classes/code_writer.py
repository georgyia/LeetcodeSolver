from classes.problem import Problem
from tools.llama_templates import make_code_interpreter_prompt, original_prompt_template, make_instruct_prompt
from tools.helpers import make_llm_request
from aleph_alpha_client import Client


class CodeWriter:
    def __init__(self, problem: Problem, client: Client, model: str):
        self.problem = problem
        self.client = client
        self.model = model

    def create_prompt(self, problem_statement: str, cot_user_prompt : str, cot_reasoning: str):
        input_output_pairs = [f"Input: {k}, Expected Output: {v}" for k, v in zip(self.problem.input_output['inputs'], self.problem.input_output['outputs'])]
        system_prompt = f"""
        You are a world-class leet-code expert. You will be given a software engineering problem statement
        and you task is to generate python code that solves the problem.
        Don't create a tab at the beginning before `def` 
        Avoid hitting the following error: An error occurred: unhashable type: 'list'
        This is the problem statement. Produce python code that solves the problem:
        {problem_statement}
        Those are expected input-output pairs: {input_output_pairs}
        You must start with following starter code:{self.problem.starter_code} 
        """

        user_prompt = f"""
        {cot_user_prompt}
        Here is the chain of thought reasoning you did to solve this problem: {cot_reasoning}
        Taking this into account, write the python code that solves the problem
        """
        prompt = make_code_interpreter_prompt(system_prompt, user_prompt)
        return prompt

    def write_code(self, problem_statement: str, code: str, traceback: list[str], incorrect_results: dict, temp: float) -> str:
        cot_prompt, cot_user_prompt = self.get_cot_prompt(problem_statement, code, traceback, incorrect_results)
        cot_response = self.do_chain_of_thought(cot_prompt, temp)
        prompt = self.create_prompt(problem_statement, cot_user_prompt, cot_response)
        code = make_llm_request(prompt, self.client, self.model, temp)
        return code

    def do_chain_of_thought(self, cot_prompt: str, temp: float):
        return make_llm_request(cot_prompt, self.client, self.model, temp)

    def get_cot_prompt(self, problem_statement: str, code: str, traceback: list[str], incorrect_results: dict):
        input_output_pairs = [f"Input: {k}, Expected Output: {v}" for k, v in zip(self.problem.input_output['inputs'], self.problem.input_output['outputs'])]
        system_prompt = f"""
        You are a world-class leet-code expert.
        You will be given a software engineering problem statement
        and you task is to analyse it and break down how would you approach solving it and explain your reasoning step by step.
        This is the problem statement:
        {problem_statement}
        Those are expected input-output pairs: {input_output_pairs}
        """
        user_prompt = ""
        if code:
            user_prompt += f"""
            You already attempted this task and provided following code: {code}
            """
        if traceback:
            user_prompt += f"""
            However, it didn't execute successfully and failed with the following traceback: {traceback}
            Explain your reasoning step by step why it failed and what changes you would make to the code to fix it
            """
        if incorrect_results:
            incorrect_list = [f"Input: {k}, Expected Output: {v}" for k, v in incorrect_results.items()]
            user_prompt += f""""
            It also failed for some test cases. Namely, those are the required input-output pairs that you fail on now
            {incorrect_list}
            Explain your reasoning step by step why it failed and what changes you would make to the code to fix it
            """

        prompt = make_instruct_prompt(system_prompt, user_prompt)
        return prompt, user_prompt

    def write_code_original_template(self, problem_dict: dict):
        prompt = original_prompt_template(problem_dict)
        return make_llm_request(prompt, self.client, self.model)
