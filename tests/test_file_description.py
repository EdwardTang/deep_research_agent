"""
Test script for file_description functionality.
"""

import os
import sys
import time
import shutil
import zipfile
import pytest
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import get_file_description

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Constants
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")
SCRATCHPAD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratchpad.md")

@pytest.fixture(scope="module")
def test_files():
    """Create test files for different formats."""
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    
    # Create a test text file
    text_path = os.path.join(TEST_FILES_DIR, "test.txt")
    with open(text_path, "w") as f:
        f.write("This is a test document.\nIt contains multiple lines.\nLine three has some numbers: 123, 456.")
    
    # Create a test image file
    image_path = os.path.join(TEST_FILES_DIR, "test.png")
    if not os.path.exists(image_path):
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
    
    # Create a test zip file
    zip_path = os.path.join(TEST_FILES_DIR, "test.zip")
    with zipfile.ZipFile(zip_path, 'w') as zip_ref:
        zip_ref.write(text_path, os.path.basename(text_path))
        zip_ref.write(image_path, os.path.basename(image_path))
    
    yield text_path, image_path, zip_path
    
    # Cleanup after tests
    shutil.rmtree(TEST_FILES_DIR)

def test_text_file_description(test_files):
    """Test description generation for text files."""
    text_path, _, _ = test_files
    result = get_file_description(text_path, "What numbers are mentioned in the text?")
    assert result is not None
    assert "123" in result and "456" in result
    assert "Error" not in result

def test_image_file_description(test_files):
    """Test description generation for image files."""
    _, image_path, _ = test_files
    result = get_file_description(image_path, "What color is dominant in this image?")
    assert result is not None
    assert "red" in result.lower()
    assert "Error" not in result

def test_zip_file_description(test_files):
    """Test description generation for zip files."""
    _, _, zip_path = test_files
    result = get_file_description(zip_path, extract_archives=True)
    assert result is not None
    assert "Archive contents:" in result
    assert "test.txt" in result
    assert "test.png" in result
    assert "Error" not in result

def test_error_handling():
    """Test error handling for nonexistent files."""
    result = get_file_description(os.path.join(TEST_FILES_DIR, "nonexistent.txt"))
    assert "Error: File not found" in result

def main():
    """Run all test cases and report results."""
    # Create test files
    text_path, image_path, zip_path = create_test_files()
    
    results: Dict[str, Dict] = {}
    
    # Test Case 1: Text File
    results["text_file"] = {
        "name": "Text File Description",
        "file": text_path,
        "question": "What numbers are mentioned in the text?"
    }
    
    # Test Case 2: Image File
    results["image_file"] = {
        "name": "Image File Description",
        "file": image_path,
        "question": "What color is dominant in this image?"
    }
    
    # Test Case 3: Zip File
    results["zip_file"] = {
        "name": "Zip File Description",
        "file": zip_path,
        "question": None,
        "extract_archives": True
    }
    
    # Test Case 4: Error Handling
    results["error_handling"] = {
        "name": "Error Handling",
        "file": os.path.join(TEST_FILES_DIR, "nonexistent.txt"),
        "question": None
    }
    
    # Run tests and collect results
    print("\nStarting file description integration tests...")
    print(f"Using test files from: {TEST_FILES_DIR}")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = 0
    total_time = 0
    
    for test_id, test_case in results.items():
        success, output, response_time = run_test_case(
            test_case["name"],
            test_case["file"],
            test_case.get("question"),
            test_case.get("extract_archives", True)
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
        f.write("\n\n## Latest File Description Test Results\n")
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