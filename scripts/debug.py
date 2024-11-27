import json

from aleph_alpha_client import Client, CompletionRequest, Prompt
from utilities import load_sample, run_test_cases, score
from tools.llama_templates import make_tool_calling_prompt, make_instruct_prompt, make_code_interpreter_prompt
from classes.problem import Problem
from classes.shit_filter import ShitFilter
from classes.code_writer import CodeWriter
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

AA_TOKEN=os.getenv("AA_TOKEN")
CLIENT = Client(AA_TOKEN)
MODEL = "llama-3.1-70b-instruct-long-context"
failed_problems = {}


def save_json(d: dict):
    current_datetime = str(datetime.now())
    with open("data/failed_tests/" + current_datetime + '.json') as f:
        json.dump(d, f)


def generate_code(problem: dict, client: Client) -> str:
    """
    Implement the generation function.

    Args:
        problem (dict): The problem to solve.

    Returns:
        str: The generated code.
    """

    problem_obj = Problem(**problem)
    # filtered_shit = sf.filter()
    sf = ShitFilter(problem_obj, CLIENT, MODEL)
    cw = CodeWriter(problem_obj, CLIENT, MODEL)
    # code = cw.write_code_original_template(problem)

    code = None
    traces = []
    incorrect_results = {}
    is_all_tests_passed = False
    for i in range(4):
        temp = 0.3
        for _ in range(8):
            # filtered_shit = sf.filter()
            code = cw.write_code(problem_obj.question, code, traces, incorrect_results, temp)

            # Run test cases on the generated code and iterate.
            test_results = run_test_cases(
               problem=problem,
               generation=code,
               timeout=10,
            )

            for result in test_results:
                if not result['passed']:
                    if result['traceback']:
                        print(f"Syntax Error for {problem_obj.problem_id}")
                        traces.append(result['traceback'])
                    else:
                        if str(result['input']) in incorrect_results and result['output'] in incorrect_results.values():
                            print(f"Same incorrect results! {problem_obj.problem_id}")
                        else:
                            print(f"Incorrect Result {problem_obj.problem_id}")
                        incorrect_results[str(result['input'])] = result['output']

            is_all_tests_passed = all([test_result['passed'] for test_result in test_results])
            if is_all_tests_passed:
                print(f"Test Passed!")
                break

        if is_all_tests_passed:
            break

    if not is_all_tests_passed:
        print(f"Failed to pass all tests for {problem_obj.problem_id}")
        failed_problems[problem_obj.problem_id] = {
            'code': code,
            'traces': traces,
            'incorrect_results': incorrect_results,
            'difficulty': problem_obj.difficulty
        }

    return code


if __name__ == '__main__':
    # problem = load_sample(index=2, dataset_path="./data/dev")
    # generate_code(problem, CLIENT)
    passed_problems, passed_test_cases = score(
        generation_func=generate_code,
        client=CLIENT,
        dataset_path="./data/val",
        length=100,
    )
    save_json(failed_problems)
    print(f"Passed Problems: {passed_problems}, \n Passed Test Cases: {passed_test_cases}")