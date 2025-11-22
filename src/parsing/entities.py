import re
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 1. EXPANDED Dictionaries for matchers
# -----------------------------

# EXPANDED Programming Languages
PROGRAMMING_LANGUAGES = [
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", 
    "php", "rust", "scala", "swift", "objective-c", "kotlin", "r", "matlab",
    "html", "css", "node.js", "sql", "bash", "shell", "powershell"
]

# EXPANDED Frameworks (Now includes Modern AI/ML libs)
FRAMEWORKS = [
    "django", "flask", "spring boot", "express", "react", "angular", "vue", 
    "rails", "fastapi", "express.js", "react.js", "vue.js", "angular.js",
    "django rest framework", "spring cloud", "react native", "flutter", 
    "redux", "vuex", "svelte", "next.js", "nuxt.js",
    # AI / ML Frameworks
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
    "matplotlib", "seaborn", "xgboost", "opencv", "huggingface", "transformers",
    "langchain", "llamaindex", "openai api"
]

# EXPANDED Databases
DATABASES = [
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "mariadb",
    "memcached", "elasticsearch", "cassandra", "dynamodb", "firestore", "couchdb"
]

# EXPANDED Cloud platforms
CLOUDS = [
    "aws", "azure", "gcp", "google cloud", "digitalocean", "heroku",
    "ec2", "rds", "s3", "lambda", "kubernetes", "docker", "jenkins", "gitlab ci"
]

# EXPANDED Tools (Includes AI Concepts & Dev Tools)
TOOLS = [
    "git", "github", "gitlab", "jira", "terraform", "ansible", "linux",
    "kafka", "rabbitmq", "spark", "hadoop", "airflow", "tableau", "power bi",
    "excel", "figma", "photoshop", "illustrator",
    # Modern AI Concepts (Great for Keywords)
    "generative ai", "llm", "large language models", "rag", "fine-tuning",
    "prompt engineering", "computer vision", "nlp", "sentiment analysis",
    "adobe creative suite", "branding", "ui/ux", "user experience", 
    "intellectual property", "corporate law", "legal", 
    "mixing", "mastering", "pro tools", "sound engineering",
    "wireshark", "penetration testing", "network security",
    "strategic planning", "artificial intelligence", "cloud computing",
    "genomics", "pcr", "data analysis"
]

# EXPANDED Job Titles
JOB_TITLES = [
    "software engineer", "backend developer", "frontend developer", "fullstack developer",
    "data scientist", "ml engineer", "devops engineer", "project manager",
    "python developer", "java developer", "web developer", "mobile developer",
    "ai engineer", "data analyst", "system administrator", "product manager"
]

# -----------------------------
# 2. Regex Patterns (IMPROVED)
# -----------------------------
EMAIL_REGEX = r"\b[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

# Fixed: Won't match dates like 2020-2021
PHONE_REGEX = r"(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}"

# NEW: Added Degree Regex (Used by Hybrid Script)
DEGREE_REGEX = r"(?i)\b(b\.?s\.?|m\.?s\.?|ph\.?d|mba|bachelor|master|doctorate)\b.{0,20}"

# -----------------------------
# 3. Setup spaCy PhraseMatcher
# -----------------------------
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

def add_patterns(label, phrases):
    patterns = [nlp.make_doc(str(p)) for p in phrases]
    matcher.add(label, patterns)

add_patterns("PROGRAMMING_LANGUAGE", PROGRAMMING_LANGUAGES)
add_patterns("FRAMEWORK", FRAMEWORKS)
add_patterns("DATABASE", DATABASES)
add_patterns("CLOUD", CLOUDS)
add_patterns("TOOL", TOOLS)
add_patterns("JOB_TITLE", JOB_TITLES)

# -----------------------------
# 4. Extraction Function (Backward Compatible)
# -----------------------------
def extract_entities(text):
    doc = nlp(text)

    entities = {
        "EMAIL": re.findall(EMAIL_REGEX, text),
        "PHONE": re.findall(PHONE_REGEX, text),
        # Added DEGREE here - CRF usually ignores extra keys, so this is safe.
        # If CRF crashes, remove this line and use re.findall in hybrid_parser instead.
        "DEGREE": [m.group(0).strip(" .,") for m in re.finditer(DEGREE_REGEX, text)],
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
    entities["DEGREE"] = list(set(entities["DEGREE"]))
    for k in entities["SKILLS"]:
        entities["SKILLS"][k] = list(set(entities["SKILLS"][k]))

    return entities

if __name__ == "__main__":
    # Test to ensure it works
    sample = "John Doe is a Python Developer with a B.S. in CS. Email: test@test.com"
    print(extract_entities(sample))