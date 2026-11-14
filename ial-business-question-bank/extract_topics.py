#!/usr/bin/env python3
"""
Extract the IAL Business sub-topic taxonomy from the syllabus PDF into a CSV.

Reads "IAL Business Syllabus.pdf", separates the left (name) column from the
indented learning points using x-coordinates, and writes data/topics.csv:

    unit,topic_code,topic_name
    1,1.3.1.1,The market
    ...

Run: .venv/bin/python extract_topics.py
"""

import csv
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "IAL Business Syllabus.pdf"
OUT = ROOT / "data" / "topics.csv"

TOPIC_RE = re.compile(r"^(\d)\.3\.(\d+)\s+(.+)$")
SUB_RE = re.compile(r"^(\d+)\s+(.+)$")
UNIT_CONTENT_RE = re.compile(r"^(\d)\.3\s+Unit content$")


def left_lines(page):
    """Reconstruct the left (name) column of a page, dropping indented learning points."""
    words = [w for w in page.extract_words() if w["x0"] < 180]
    rows = {}
    for w in words:
        rows.setdefault(round(w["top"] / 6) * 6, []).append(w)
    out = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w["x0"])
        line = " ".join(w["text"] for w in ws)
        if re.search(r"Pearson|Edexcel|Specification|©", line):
            continue
        if re.fullmatch(r"\d+", line.strip()):
            continue
        out.append(line)
    return out


def main():
    topics = {}          # topic_code -> name
    content_pages = []

    with pdfplumber.open(PDF) as pdf:
        for pi in range(14, 39):  # pages 15..39
            text = pdf.pages[pi].extract_text() or ""
            found = False
            for line in text.splitlines():
                m = TOPIC_RE.match(line.strip())
                if m:
                    found = True
                    code = f"{m.group(1)}.3.{m.group(2)}"
                    name = re.sub(r"\s*\(continued\)\s*$", "", m.group(3)).strip()
                    topics.setdefault(code, name)
            if found:
                content_pages.append(pi)

    # fix the one known syllabus typo
    if topics.get("3.3.6") == "Manging change":
        topics["3.3.6"] = "Managing change"

    subs = []  # (unit, sub_code, name)
    with pdfplumber.open(PDF) as pdf:
        lines = []
        for pi in content_pages:
            lines.extend(left_lines(pdf.pages[pi]))

    cur_topic = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "What students need to learn:" or UNIT_CONTENT_RE.match(line):
            i += 1
            continue
        m = TOPIC_RE.match(line)
        if m:
            cur_topic = f"{m.group(1)}.3.{m.group(2)}"
            i += 1
            continue
        m = SUB_RE.match(line)
        if m and cur_topic:
            num = int(m.group(1))
            unit = int(cur_topic.split(".")[0])
            parts = [m.group(2).strip()]
            i += 1
            while i < len(lines):
                nl = lines[i].strip()
                if not nl or nl == "What students need to learn:":
                    i += 1
                    continue
                if TOPIC_RE.match(nl) or SUB_RE.match(nl) or UNIT_CONTENT_RE.match(nl):
                    break
                parts.append(nl)
                i += 1
            subs.append((unit, f"{cur_topic}.{num}", " ".join(parts)))
            continue
        i += 1

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit", "topic_code", "topic_name"])
        w.writeheader()
        w.writerows({"unit": u, "topic_code": c, "topic_name": n} for u, c, n in subs)

    print(f"Wrote {len(subs)} sub-topics to {OUT}")


if __name__ == "__main__":
    main()
