from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Callable, Tuple

from aleph_alpha_client import Client
from tqdm import tqdm

from utilities import load_sample, run_test_cases

def score(generation_func: Callable, client: Client, dataset_path: str, length: int = 200) -> Tuple[float, float]:
    """
    Score the generation function on a given test set.

    Args:
        generation_func (Callable): The generation function to score.
        dataset_path (str): The path to the test set.
        length (int): The number of problems to score.

    Returns:
        Tuple[float, float]: The percentage of problems that were solved correctly and the
                             percentage of test cases that were solved correctly.
    """

    def evaluate_problem(index: int) -> Tuple[int, int]:
        problem = load_sample(index=index, dataset_path=dataset_path)
        problem_wo_test_cases = deepcopy(problem)
        del problem_wo_test_cases["test_cases"]
        generated_code = generation_func(
            problem=problem_wo_test_cases,
            client=client
        )

        result = run_test_cases(
            problem=problem,
            generation=generated_code,
            eval=True,
            timeout=5,
        )

        passed = [r["passed"] for r in result]
        passed_test_cases = sum(passed)
        total_test_cases = len(passed)

        passed_problems = 1 if all(passed) else 0

        return passed_problems, passed_test_cases, total_test_cases

    passed_problems = 0
    total_problems = 0
    passed_test_cases = 0
    total_test_cases = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_problem, i): i for i in range(length)}

        for future in tqdm(as_completed(futures), total=length):
            try:
                problem_result, test_cases_passed, test_cases_total = future.result()
                passed_problems += problem_result
                passed_test_cases += test_cases_passed
                total_test_cases += test_cases_total
                total_problems += 1
                print(f"\nPassed problems: {passed_problems} out of {total_problems} ({100 * passed_problems / total_problems: .0f}% accuracy) \n"
                      f"Passed test cases: {passed_test_cases} out of {total_test_cases} ({100 * passed_test_cases / total_test_cases: .0f}% accuracy)")
            except Exception as e:
                print(f"An error occurred: {e}")
                total_problems += 1

    return (
        passed_problems / length,
        passed_test_cases / total_test_cases if total_test_cases > 0 else 0.0,
    )