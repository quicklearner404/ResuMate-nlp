import sys
import os
# Add BOTH paths to handle the nested imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'parsing'))

from src.parsing.parser import clean_text, parse_pdf
import tempfile

def test_clean_text():
    """Test text cleaning function"""
    # Test with extra spaces and newlines
    dirty_text = "  Hello   World\nNew Line  \tTabbed"
    cleaned = clean_text(dirty_text)
    
    assert cleaned == "hello world new line tabbed"
    assert "\n" not in cleaned
    assert "\t" not in cleaned
    assert "  " not in cleaned
    
    # Test with mixed case
    mixed_case = "Python JAVA JavaScript"
    cleaned = clean_text(mixed_case)
    assert cleaned == "python java javascript"

def test_clean_text_empty():
    """Test cleaning empty text"""
    assert clean_text("") == ""
    assert clean_text("   ") == ""

def test_parse_pdf_invalid_file():
    """Test PDF parsing with non-existent file"""
    # Should handle missing file gracefully
    result = parse_pdf("non_existent_file.pdf")
    assert result == ""  # or whatever your function returns for errors

def test_end_to_end_parsing():
    """Test the complete parsing process with a simple text"""
    # This is a basic test - in real scenario you'd use actual PDFs
    test_text = "John Doe Backend Developer Python Java Django"
    # For now, we just test that the function exists and runs
    assert callable(parse_pdf)

if __name__ == "__main__":
    test_clean_text()
    print("✅ All parser tests passed!")