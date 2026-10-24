"""Secret scanner for repository files before GitHub publish."""

import os
import re
import sys

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|private[_-]?key|token|auth[_-]?token)\s*[:=]\s*['\"][0-9a-zA-Z\-_]{8,}['\"]"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"ghp_[0-9a-zA-Z]{36}"),
    re.compile(r"github_pat_[0-9a-zA-Z_]{82}"),
]

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "dist",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "test-results",
    "playwright-report",
    "data/raw",
    "raw",
    "docs/design",
}

BINARY_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".woff",
    ".woff2",
    ".parquet",
    ".gitkeep",
}


def scan():
    hits = []
    scanned_count = 0
    for root, dirs, files in os.walk("."):
        norm_root = root.replace("\\", "/")
        if any(ex in norm_root for ex in EXCLUDED_DIRS):
            continue
        for f in files:
            if f.startswith(".env"):
                hits.append((os.path.join(root, f), 0, "Found .env file"))
            _, ext = os.path.splitext(f)
            if ext.lower() in BINARY_EXTENSIONS:
                continue
            path = os.path.join(root, f)
            scanned_count += 1
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    for idx, line in enumerate(fp, 1):
                        for pat in SECRET_PATTERNS:
                            if pat.search(line):
                                hits.append((path, idx, line.strip()[:100]))
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error reading {path}: {e}")

    print(f"Scanned {scanned_count} files.")
    print(f"Total secret scan hits: {len(hits)}")
    for h in hits:
        print(f"  {h[0]}:{h[1]}: {h[2]}")
    return len(hits)


if __name__ == "__main__":
    sys.exit(scan())
