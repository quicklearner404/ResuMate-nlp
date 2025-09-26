import re
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 1. Dictionaries for matchers
# -----------------------------
PROGRAMMING_LANGUAGES = ["python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", "php", "rust", "scala"]
FRAMEWORKS = ["django", "flask", "spring boot", "express", "react", "angular", "vue", "rails", "fastapi"]
DATABASES = ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "mariadb"]
CLOUDS = ["aws", "azure", "gcp", "google cloud", "digitalocean", "heroku"]
TOOLS = ["docker", "kubernetes", "git", "jenkins", "terraform", "ansible", "linux"]

JOB_TITLES = ["software engineer", "backend developer", "frontend developer", "fullstack developer",
              "data scientist", "ml engineer", "devops engineer", "project manager"]

# -----------------------------
# 2. Regex Patterns
# -----------------------------
EMAIL_REGEX = r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+"
PHONE_REGEX = r"(\+?\d{1,3}[-\s]?)?\(?\d{2,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}"

# -----------------------------
# 3. Setup spaCy PhraseMatcher
# -----------------------------
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

def add_patterns(label, phrases):
    patterns = [nlp.make_doc(p) for p in phrases]
    matcher.add(label, patterns)

add_patterns("PROGRAMMING_LANGUAGE", PROGRAMMING_LANGUAGES)
add_patterns("FRAMEWORK", FRAMEWORKS)
add_patterns("DATABASE", DATABASES)
add_patterns("CLOUD", CLOUDS)
add_patterns("TOOL", TOOLS)
add_patterns("JOB_TITLE", JOB_TITLES)

# -----------------------------
# 4. Extraction Function
# -----------------------------
def extract_entities(text):
    doc = nlp(text)

    entities = {
        "EMAIL": re.findall(EMAIL_REGEX, text),
        "PHONE": re.findall(PHONE_REGEX, text),
        "JOB_TITLE": [],
        "SKILLS": {
            "PROGRAMMING_LANGUAGE": [],
            "FRAMEWORK": [],
            "DATABASE": [],
            "CLOUD": [],
            "TOOL": []
        }
    }

    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span_text = doc[start:end].text.lower()
        if label == "JOB_TITLE":
            entities["JOB_TITLE"].append(span_text)
        else:
            entities["SKILLS"][label].append(span_text)

    # Deduplicate
    entities["JOB_TITLE"] = list(set(entities["JOB_TITLE"]))
    for k in entities["SKILLS"]:
        entities["SKILLS"][k] = list(set(entities["SKILLS"][k]))

    return entities

# -----------------------------
# Debug Run
# -----------------------------
if __name__ == "__main__":
    sample = """
    John Doe
    Email: johndoe@gmail.com | Phone: +1-555-123-4567
    Backend Developer with experience in Python, Java, Django, Flask, PostgreSQL, and AWS.
    Worked on microservices with Docker and Kubernetes.
    """
    print(extract_entities(sample))
