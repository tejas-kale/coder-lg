import json
import sys
from typing import List, Dict, Any
import math

def is_close_enough(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    Compare two values with tolerance for floating point numbers.
    
    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
    
    Returns:
        bool: True if values are equal or close enough for floats
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b

def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executes test cases against the provided code.
    
    Args:
        code: String containing the Python code to test
        test_cases: List of test cases, each with input_args, expected, and description
    
    Returns:
        List of test results with execution details
    """
    # Execute the provided code in a new namespace
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return [{
            'error': f"Code execution failed: {str(e)}",
            'passed': False
        }]
    
    # Get the function to test
    if 'add' not in namespace:
        return [{
            'error': "Function 'add' not found in code",
            'passed': False
        }]
    
    add_func = namespace['add']
    results = []
    
    # Run each test case
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

def main():
    """
    Main function to run when script is executed directly.
    Expects code and test cases as command line arguments.
    """
    if len(sys.argv) != 3:
        print("Usage: python test_runner.py <code_file> <test_cases_file>")
        sys.exit(1)
    
    try:
        # Read code and test cases from files
        with open(sys.argv[1], 'r') as f:
            code = f.read()
        
        with open(sys.argv[2], 'r') as f:
            test_cases = json.load(f)
        
        # Run tests and output results
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