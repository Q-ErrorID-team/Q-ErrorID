import re
from pathlib import Path


def test_credentials_are_not_committed():
    root = Path(__file__).resolve().parents[2]
    suspicious = re.compile(r"(?:hq|haiqu)[-_][A-Za-z0-9]{24,}", re.IGNORECASE)
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache"}
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or excluded.intersection(path.parts):
            continue
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".toml",
            ".txt",
            ".example",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if suspicious.search(text):
            findings.append(str(path.relative_to(root)))
    assert findings == []
