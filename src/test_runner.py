"""Test runner module for executing code tests in a safe environment."""

import json
import sys
from typing import List, Dict, Any
import math

def is_close_enough(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    Compare two values with tolerance for floating point numbers.
    
    Args:
        a: First value to compare
        b: Second value to compare
        rel_tol: Relative tolerance for floating point comparison (default: 1e-9)
        abs_tol: Absolute tolerance for floating point comparison (default: 0.0)
    
    Returns:
        bool: True if values are equal or within tolerance for floats
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b

def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute test cases against the provided code in a clean namespace.
    
    Args:
        code: String containing the Python code to test
        test_cases: List of test case dictionaries, each containing:
            - input_args: List of arguments to pass to the function
            - expected: Expected output value
            - description: Description of the test case
    
    Returns:
        List[Dict[str, Any]]: List of test results, each containing:
            - input: String representation of the function call
            - output: Actual output from the function (if successful)
            - expected: Expected output value
            - passed: Boolean indicating if test passed
            - description: Description of the test case
            - error: Error message (if test failed)
    
    Raises:
        Exception: If code execution fails or function is not found
    """
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return [{
            'error': f"Code execution failed: {str(e)}",
            'passed': False
        }]
    
    if 'add' not in namespace:
        return [{
            'error': "Function 'add' not found in code",
            'passed': False
        }]
    
    add_func = namespace['add']
    results = []
    
    for test in test_cases:
        try:
            result = add_func(*test['input_args'])
            passed = is_close_enough(result, test['expected'])
            results.append({
                'input': f"add{tuple(test['input_args'])}",
                'output': result,
                'expected': test['expected'],
                'passed': passed,
                'description': test['description']
            })
        except Exception as e:
            results.append({
                'input': f"add{tuple(test['input_args'])}",
                'error': str(e),
                'passed': False,
                'description': test['description']
            })
    
    return results

def main() -> None:
    """
    Main entry point for the test runner script.
    
    Expects two command-line arguments:
        1. Path to file containing code to test
        2. Path to file containing test cases in JSON format
    
    Prints:
        JSON string containing test results
    
    Returns:
        None
    
    Raises:
        SystemExit: If incorrect number of arguments or execution fails
    """
    if len(sys.argv) != 3:
        print("Usage: python test_runner.py <code_file> <test_cases_file>")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], 'r') as f:
            code = f.read()
        
        with open(sys.argv[2], 'r') as f:
            test_cases = json.load(f)
        
        results = run_tests(code, test_cases)
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps([{
            'error': f"Test runner error: {str(e)}",
            'passed': False
        }]))
        sys.exit(1)

if __name__ == "__main__":
    main() 