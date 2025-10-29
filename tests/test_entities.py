import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parsing.entities import extract_entities

def test_extract_programming_languages():
    """Test extraction of programming languages"""
    text = "I am proficient in Python, Java, and JavaScript for backend development."
    result = extract_entities(text)
    
    programming_langs = result["SKILLS"]["programming_languages"]
    assert "python" in programming_langs
    assert "java" in programming_langs
    assert "javascript" in programming_langs

def test_extract_frameworks():
    """Test extraction of frameworks"""
    text = "Experience with Django, Flask, and React for web development."
    result = extract_entities(text)
    
    frameworks = result["SKILLS"]["frameworks"]
    assert "django" in frameworks
    assert "flask" in frameworks
    assert "react" in frameworks

def test_extract_job_titles():
    """Test job title extraction"""
    text = "Backend Developer with experience as a Data Scientist and ML Engineer."
    result = extract_entities(text)
    
    job_titles = result["JOB_TITLE"]
    assert "backend developer" in job_titles
    assert "data scientist" in job_titles
    assert "ml engineer" in job_titles

def test_extract_databases():
    """Test database extraction"""
    text = "Worked with MySQL, MongoDB, and PostgreSQL databases."
    result = extract_entities(text)
    
    databases = result["SKILLS"]["databases"]
    assert "mysql" in databases
    assert "mongodb" in databases
    assert "postgresql" in databases

def test_extract_cloud_tools():
    """Test cloud and tools extraction"""
    text = "Deployed applications on AWS using Docker and Kubernetes."
    result = extract_entities(text)
    
    assert "aws" in result["SKILLS"]["cloud"]
    assert "docker" in result["SKILLS"]["tools"]
    assert "kubernetes" in result["SKILLS"]["tools"]

def test_no_false_positives():
    """Test that common words aren't incorrectly extracted as skills"""
    text = "I like to drink java coffee and play with my pet python."
    result = extract_entities(text)
    
    # Should not extract "java" or "python" as programming languages in this context
    programming_langs = result["SKILLS"]["programming_languages"]
    assert len(programming_langs) == 0

def test_case_insensitivity():
    """Test that extraction works regardless of case"""
    text = "Skills: PYTHON, Java, DJANGO, REACT"
    result = extract_entities(text)
    
    programming_langs = result["SKILLS"]["programming_languages"]
    frameworks = result["SKILLS"]["frameworks"]
    
    assert "python" in programming_langs
    assert "java" in programming_langs
    assert "django" in frameworks
    assert "react" in frameworks

if __name__ == "__main__":
    # Run basic test
    test_text = "Python developer with Django and PostgreSQL experience."
    result = extract_entities(test_text)
    print("✅ Basic entity extraction test:")
    print(f"   Skills: {result['SKILLS']}")
    print(f"   Job Titles: {result['JOB_TITLE']}")