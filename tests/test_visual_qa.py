"""
Test script for visual_qa functionality.
"""

import os
import sys
import time
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import visual_qa

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Constants
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")
SCRATCHPAD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratchpad.md")

def run_test_case(test_name: str, image_path: str, question: Optional[str] = None) -> Tuple[bool, str, float]:
    """
    Run a single test case and return results.
    
    Args:
        test_name: Name of the test case
        image_path: Path to test image
        question: Optional question about the image
        
    Returns:
        Tuple of (success, output, response_time)
    """
    print(f"\nRunning test: {test_name}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result = visual_qa(image_path, question)
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
    results: Dict[str, Dict] = {}
    
    # Test Case 1: Basic Image Description
    results["basic_description"] = {
        "name": "Basic Image Description",
        "image": os.path.join(TEST_IMAGES_DIR, "test1.jpg"),
        "question": None
    }
    
    # Test Case 2: Specific Question
    results["specific_question"] = {
        "name": "Specific Question",
        "image": os.path.join(TEST_IMAGES_DIR, "test1.jpg"),
        "question": "What colors are present in this image?"
    }
    
    # Test Case 3: Error Handling
    results["error_handling"] = {
        "name": "Error Handling",
        "image": os.path.join(TEST_IMAGES_DIR, "nonexistent.jpg"),
        "question": None
    }
    
    # Run tests and collect results
    print("\nStarting visual_qa integration tests...")
    print(f"Using test images from: {TEST_IMAGES_DIR}")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = 0
    total_time = 0
    
    for test_id, test_case in results.items():
        success, output, response_time = run_test_case(
            test_case["name"],
            test_case["image"],
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
        f.write("\n\n## Latest Test Results\n")
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