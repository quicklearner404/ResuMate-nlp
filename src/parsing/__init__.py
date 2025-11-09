# src/parsing/__init__.py - Optional: make imports cleaner
from .hybrid import analyze_resume
from .entities import extract_entities
from .train_crf import get_crf_model

# Now you can do: from src.parsing import analyze_resume
# Instead of: from src.parsing.hybrid import analyze_resume