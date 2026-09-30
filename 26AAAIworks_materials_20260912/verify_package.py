#!/usr/bin/env python3
"""Verify the extracted material package using only the Python standard library."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote, urlsplit

def verify(root):
    manifest = json.loads((root / "文件清单.json").read_text())
    issues = []
    for name, entry in manifest["files"].items():
        path = root / name
        if not path.is_file():
            issues.append("Missing file: " + name)
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != entry["sha256"]:
            issues.append("SHA-256 mismatch: " + name)
    links = 0
    for path in root.rglob("*.md"):
        text = re.sub(r"```.*?```|~~~.*?~~~", "", path.read_text(), flags=re.S)
        for m in re.finditer(r"\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text):
            u = urlsplit(m[1].strip("<>"))
            if u.scheme or u.netloc or not u.path:
                continue
            links += 1
            target = (path.parent / unquote(u.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                issues.append(str(path.relative_to(root)) + ": invalid local link " + u.path)
    return {"passed": not issues, "files_checked": len(manifest["files"]),
            "local_links_checked": links, "issues": issues,
            "scope": "File hashes and local Markdown path targets; external citations and heading anchors are not checked."}

if __name__ == "__main__":
    report = verify(Path(__file__).resolve().parent)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
