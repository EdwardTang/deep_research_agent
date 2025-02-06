"""
Test script for targeted description functionality.
"""

import os
import sys
import pytest
from PIL import Image
from dotenv import load_dotenv

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import generate_targeted_description

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Constants
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")

@pytest.fixture(scope="module")
def test_files():
    """Create test files for different formats."""
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    
    # Create a test text file
    text_path = os.path.join(TEST_FILES_DIR, "test_doc.txt")
    with open(text_path, "w") as f:
        f.write("""The red car is parked in front of a blue house.
The house has a large oak tree in the yard.
There are three children playing in the garden.
The sky is cloudy and it looks like it might rain.
The car's license plate reads XYZ-123.""")
    
    # Create a test image file
    image_path = os.path.join(TEST_FILES_DIR, "test_img.png")
    if not os.path.exists(image_path):
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
    
    yield text_path, image_path
    
    # Cleanup after tests
    import shutil
    shutil.rmtree(TEST_FILES_DIR)

def test_document_description(test_files):
    """Test targeted description generation for documents."""
    text_path, _ = test_files
    
    # Test with a question about numbers
    result = generate_targeted_description(
        text_path,
        "What numbers appear in the text?",
        "document"
    )
    assert result is not None
    assert "123" in result
    assert "license plate" in result.lower()
    assert "Error" not in result
    
    # Test with a question about colors
    result = generate_targeted_description(
        text_path,
        "What colors are mentioned?",
        "document"
    )
    assert result is not None
    assert "red" in result.lower()
    assert "blue" in result.lower()
    assert "Error" not in result

def test_image_description(test_files):
    """Test targeted description generation for images."""
    _, image_path = test_files
    
    # Test with a question about colors
    result = generate_targeted_description(
        image_path,
        "What colors are present in the image?",
        "image"
    )
    assert result is not None
    # Check for color-related terms
    assert any(term in result.lower() for term in ["color", "hue", "shade", "vivid", "bold"])
    assert "Error" not in result
    
    # Test with a question about objects
    result = generate_targeted_description(
        image_path,
        "What objects can you see in the image?",
        "image"
    )
    assert result is not None
    # Check for shape-related terms
    assert any(term in result.lower() for term in ["square", "rectangle", "shape", "geometric"])
    assert "Error" not in result

def test_error_handling():
    """Test error handling for nonexistent files."""
    result = generate_targeted_description(
        os.path.join(TEST_FILES_DIR, "nonexistent.txt"),
        "What is in the file?"
    )
    assert "Error: File not found" in result 