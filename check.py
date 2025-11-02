import os

path = os.path.join(os.path.dirname(__file__), "data/elderly_guidelines.txt")
print(os.path.exists(path))  # Should print True
with open(path, "r", encoding="utf-8") as f:
    print(f.read()[:200])  # Print first 200 chars
