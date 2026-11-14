#!/usr/bin/env python3
"""
Phase 3 — taxonomy + question tagging loader.

Populates the `topics` table from a CSV of syllabus sub-topics, and the
`question_topics` join table from a tagging CSV. Idempotent.

Usage (from the project root):

    # 1) load the sub-topic taxonomy (already extracted to data/topics.csv)
    .venv/bin/python tag.py --topics data/topics.csv

    # 2) load question -> topic mappings
    .venv/bin/python tag.py --tags data/tags.csv

CSV formats
-----------
topics.csv  (one row per sub-topic):
    unit,topic_code,topic_name
    1,1.3.1.1,The market
    1,1.3.1.2,Market research

tags.csv    (one row per question->sub-topic link; question_number is the DB label):
    paper_id,question_number,topic_code
    2023-unit3-2023-06-WBS13-01,1(a),1.3.1.1
    2023-unit3-2023-06-WBS13-01,1(b),1.3.2.4
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "qbank.db"


def conn():
    c = sqlite3.connect(DB)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def load_topics(path: Path):
    c = conn()
    n = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            unit = int(row["unit"])
            code = row["topic_code"].strip()
            name = row["topic_name"].strip()
            if not code or not name:
                print(f"  skipping row {i} (empty code/name)")
                continue
            c.execute(
                """INSERT INTO topics (unit, topic_code, topic_name, sort_order)
                   VALUES (?,?,?,?)
                   ON CONFLICT(topic_code) DO UPDATE SET
                     unit=excluded.unit, topic_name=excluded.topic_name,
                     sort_order=excluded.sort_order""",
                (unit, code, name, i),
            )
            n += 1
    c.commit()
    c.close()
    print(f"Loaded {n} sub-topics into `topics`.")


def load_tags(path: Path):
    c = conn()
    n = 0
    missing_q = []
    missing_t = []
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            paper_id = row["paper_id"].strip()
            qnum = row["question_number"].strip()
            code = row["topic_code"].strip()

            q = c.execute(
                "SELECT question_id FROM questions WHERE paper_id=? AND question_number=?",
                (paper_id, qnum),
            ).fetchone()
            t = c.execute("SELECT topic_id FROM topics WHERE topic_code=?", (code,)).fetchone()

            if q is None:
                missing_q.append((i, paper_id, qnum))
                continue
            if t is None:
                missing_t.append((i, code))
                continue
            c.execute(
                "INSERT OR IGNORE INTO question_topics (question_id, topic_id) VALUES (?,?)",
                (q[0], t[0]),
            )
            n += 1
    c.commit()
    c.close()
    print(f"Linked {n} question->topic rows into `question_topics`.")
    if missing_q:
        print("  !! questions not found (row, paper_id, qnum):")
        for x in missing_q:
            print("    ", x)
    if missing_t:
        print("  !! topic codes not found (row, code):")
        for x in missing_t:
            print("    ", x)


def main():
    ap = argparse.ArgumentParser(description="Load taxonomy + question tags into SQLite.")
    ap.add_argument("--topics", type=str, default=None, help="topics CSV")
    ap.add_argument("--tags", type=str, default=None, help="question->topic tags CSV")
    ap.add_argument("--db", type=str, default=str(DB))
    args = ap.parse_args()

    if not args.topics and not args.tags:
        sys.exit("Provide --topics and/or --tags. See --help.")

    if args.topics:
        load_topics(Path(args.topics))
    if args.tags:
        load_tags(Path(args.tags))


if __name__ == "__main__":
    main()
