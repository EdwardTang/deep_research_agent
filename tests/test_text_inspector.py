"""
Test script for text_inspector functionality.
"""

import os
import sys
import time
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import inspect_file_as_text

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Constants
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")
SCRATCHPAD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratchpad.md")

def run_test_case(test_name: str, file_path: str, question: Optional[str] = None) -> Tuple[bool, str, float]:
    """
    Run a single test case and return results.
    
    Args:
        test_name: Name of the test case
        file_path: Path to test file
        question: Optional question about the file
        
    Returns:
        Tuple of (success, output, response_time)
    """
    print(f"\nRunning test: {test_name}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result = inspect_file_as_text(file_path, question)
        success = True
    except Exception as e:
        result = f"Error: {str(e)}"
        success = False
    response_time = time.time() - start_time
    
    print(f"Result: {'Success' if success else 'Failed'}")
    print(f"Response time: {response_time:.2f}s")
    print(f"Output: {result[:200]}..." if len(result) > 200 else f"Output: {result}")
    
    return success, result, response_time

def main():
    """Run all test cases and report results."""
    # Create test files directory if it doesn't exist
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    
    # Create a test text file
    test_txt_path = os.path.join(TEST_FILES_DIR, "test1.txt")
    with open(test_txt_path, "w") as f:
        f.write("This is a test document.\nIt contains multiple lines.\nLine three has some numbers: 123, 456.")
    
    results: Dict[str, Dict] = {}
    
    # Test Case 1: Basic Text Reading
    results["basic_reading"] = {
        "name": "Basic Text Reading",
        "file": test_txt_path,
        "question": None
    }
    
    # Test Case 2: Specific Question
    results["specific_question"] = {
        "name": "Specific Question",
        "file": test_txt_path,
        "question": "What numbers are mentioned in the text?"
    }
    
    # Test Case 3: Error Handling
    results["error_handling"] = {
        "name": "Error Handling",
        "file": os.path.join(TEST_FILES_DIR, "nonexistent.txt"),
        "question": None
    }
    
    # Run tests and collect results
    print("\nStarting text_inspector integration tests...")
    print(f"Using test files from: {TEST_FILES_DIR}")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = 0
    total_time = 0
    
    for test_id, test_case in results.items():
        success, output, response_time = run_test_case(
            test_case["name"],
            test_case["file"],
            test_case["question"]
        )
        
        results[test_id].update({
            "success": success,
            "output": output,
            "response_time": response_time
        })
        
        if success:
            passed_tests += 1
        total_time += response_time
    
    # Print summary
    print("\nTest Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {total_tests - passed_tests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average response time: {total_time/total_tests:.2f}s")
    
    # Update scratchpad with results
    with open(SCRATCHPAD_PATH, "a") as f:
        f.write("\n\n## Latest Text Inspector Test Results\n")
        f.write(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- Total tests: {total_tests}\n")
        f.write(f"- Passed tests: {passed_tests}\n")
        f.write(f"- Failed tests: {total_tests - passed_tests}\n")
        f.write(f"- Average response time: {total_time/total_tests:.2f}s\n\n")
        
        for test_id, test_case in results.items():
            f.write(f"### {test_case['name']}\n")
            f.write(f"- Success: {'✓' if test_case['success'] else '✗'}\n")
            f.write(f"- Response time: {test_case['response_time']:.2f}s\n")
            f.write(f"- Output: {test_case['output'][:200]}...\n\n")

if __name__ == "__main__":
    main() 