import json

with open("tech_resumes.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        resume = json.loads(line.strip())
        print("Resume content:\n", resume["content"])
        print("NER annotations:\n", resume["annotation"])
        if i == 2:  # Preview first 3 resumes
            break
