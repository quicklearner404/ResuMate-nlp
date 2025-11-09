import re
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 1. EXPANDED Dictionaries for matchers
# -----------------------------

# EXPANDED Programming Languages (added all missing ones from our analysis)
PROGRAMMING_LANGUAGES = [
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", 
    "php", "rust", "scala", "swift", "objective-c", "kotlin", "r", "matlab",
    "html", "css", "node.js", "sql"
]

# EXPANDED Frameworks (added all the missing ones we found)
FRAMEWORKS = [
    "django", "flask", "spring boot", "express", "react", "angular", "vue", 
    "rails", "fastapi", "express.js", "react.js", "vue.js", "angular.js",
    "django rest framework", "spring cloud", "gin", "sinatra", "play framework",
    "react native", "flutter", "backbone.js", "styled-components", "redux", 
    "vuex", "mobx", "svelte", "uikit", "swiftui", "android sdk", "jetpack",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
    "matplotlib", "seaborn", "xgboost", "sqlalchemy", "flask sqlalchemy",
    "alembic", "celery"
]

# EXPANDED Databases (added missing ones)
DATABASES = [
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "mariadb",
    "memcached", "elasticsearch", "cassandra", "dynamodb"
]

# EXPANDED Cloud platforms
CLOUDS = [
    "aws", "azure", "gcp", "google cloud", "digitalocean", "heroku",
    "ec2", "rds", "s3", "lambda", "cloud storage", "aws ecs", "docker swarm"
]

# EXPANDED Tools (added all the missing development tools and processes)
TOOLS = [
    "docker", "kubernetes", "git", "jenkins", "terraform", "ansible", "linux",
    "github", "gitlab", "ci/cd", "continuous integration", "continuous deployment",
    "jenkins", "gitlab ci/cd", "aws codepipeline", "pulumi", "cloudformation",
    "prometheus", "grafana", "elk stack", "rabbitmq", "kafka", "apache spark",
    "web scrapers", "data pipelines", "etl", "restful apis", "graphql", "socket.io",
    "microservices", "serverless", "oauth", "oauth 2.0", "jwt", "owasp",
    "unit tests", "jest", "mocha", "testing", "load testing", "performance testing",
    "caching", "load balancing", "indexing", "profiling", "optimization",
    "responsive design", "css-in-js", "auto layout", "constraintlayout",
    "testflight", "firebase test lab", "bluetooth integration", "calendar integration",
    "in-app purchases", "push notifications", "app store", "google play store",
    "data preprocessing", "feature engineering", "model evaluation", "model deployment",
    "exploratory data analysis", "statistical inference", "machine learning",
    "deep learning", "supervised learning", "unsupervised learning",
    "natural language processing", "computer vision", "sentiment analysis",
    "anomaly detection", "recommendation systems", "customer segmentation",
    "predictive modeling", "time series", "neural networks", "cnn", "rnn",
    "convolutional neural networks", "recurrent neural networks", "clustering",
    "regression", "classification", "feature importance", "model explainability",
    "a/b testing", "hypothesis testing", "data visualization", "data analysis",
    "data warehousing", "big data", "distributed computing"
]

# EXPANDED Job Titles (added all the missing ones we found)
JOB_TITLES = [
    "software engineer", "backend developer", "frontend developer", "fullstack developer",
    "data scientist", "ml engineer", "devops engineer", "project manager",
    "python developer", "java developer", "javascript developer", "mobile developer",
    "ios developer", "android developer", "cloud engineer", "machine learning engineer",
    "full stack developer", "mobile app developer", "web developer", "data engineer",
    "ai engineer", "front end developer", "back end developer"
]

# -----------------------------
# 2. Regex Patterns
# -----------------------------
EMAIL_REGEX = r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+"
# In entities.py, update the phone regex:
PHONE_REGEX = r"[\+]?[(]?[\d\s\-\(\)]{10,}"

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
    Worked on microservices with Docker and Kubernetes. Also knows React Native and TensorFlow.
    """
    print(extract_entities(sample))