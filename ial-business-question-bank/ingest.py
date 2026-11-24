#!/usr/bin/env python3
"""
Phase 1 ingestion — reads Pearson IAL Business past papers (+ mark schemes),
parses their Source Booklet (Extracts) and questions, and writes them into SQLite.

Usage (run from the project root, using the project virtualenv):

    # full workspace
    .venv/bin/python ingest.py --all --db data/qbank.db

    # a single paper (a directory, base stem, or -qp.pdf path)
    .venv/bin/python ingest.py --paper "IAL Business Past Papers/June 2023/Unit 3" --db data/qbank.db

Idempotent: each paper is delete-then-insert, so re-runs do not duplicate rows.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent
DEFAULT_PAPER = ROOT / "IAL Business Past Papers" / "June 2023" / "Unit 3"
DEFAULT_DB = ROOT / "data" / "qbank.db"

# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------

def resolve_paths(arg: str) -> tuple[Path, Path]:
    """Return (qp_path, ms_path) for a paper argument."""
    p = Path(arg).expanduser()
    if p.is_dir():
        qps = sorted(p.glob("*-qp.pdf"))
        mss = sorted(p.glob("*-ms.pdf"))
        if not qps or not mss:
            sys.exit(f"ERROR: directory {p} must contain one -qp.pdf and one -ms.pdf")
        return qps[0].resolve(), mss[0].resolve()
    if p.suffix == ".pdf":
        if p.name.endswith("-qp.pdf"):
            return p.resolve(), Path(str(p)[:-len("-qp.pdf")] + "-ms.pdf").resolve()
        if p.name.endswith("-ms.pdf"):
            return Path(str(p)[:-len("-ms.pdf")] + "-qp.pdf").resolve(), p.resolve()
        sys.exit("ERROR: expected a -qp.pdf or -ms.pdf filename")
    # base stem
    base = str(p)
    if base.endswith("-qp"):
        base = base[:-3]
    if base.endswith("-ms"):
        base = base[:-3]
    return Path(base + "-qp.pdf").resolve(), Path(base + "-ms.pdf").resolve()


def paper_meta(qp_path: Path) -> dict:
    """Derive stable metadata from the file path (folder + filename)."""
    rel = qp_path.relative_to(ROOT)
    parts = rel.parts  # e.g. ("IAL Business Past Papers", "June 2023", "Unit 3", "<file>")
    unit_folder = parts[-2]                      # "Unit 3"
    session_label = parts[-3]                    # "June 2023"
    unit = int(re.search(r"(\d)", unit_folder).group(1))

    stem = qp_path.name[:-len("-qp.pdf")]        # "2023-unit3-2023-06-WBS13-01"
    bits = stem.split("-")                       # [year, unitN, year, month, WBSxx, nn]
    exam_year = int(bits[0])
    exam_month = int(bits[3])
    paper_code = f"{bits[4]}-{bits[5]}"          # "WBS13-01"

    return {
        "paper_id": stem,
        "unit": unit,
        "exam_year": exam_year,
        "exam_month": exam_month,
        "session_label": session_label,
        "paper_code": paper_code,
        "paper_pdf_path": str(rel),
        "mark_scheme_pdf_path": str(rel.with_name(stem + "-ms.pdf")),
    }


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

SKIP_EXACT = {
    "Turn over", "BLANK PAGE", "Acknowledgements",
    "Do not return this Source Booklet with the question paper.",
    "Do not return this Source Booklet with the question paper",
    # vertical "DO NOT WRITE IN THIS AREA" margin text (rotated -> words on separate lines)
    "DO", "NOT", "WRITE", "IN", "THIS", "AREA",
    "AERA", "SIHT", "NI", "ETIRW", "TON", "OD",
    "AERA SIHT NI ETIRW TON OD", "SIHT NI ETIRW TON OD",
}

def clean_one_line(raw: str) -> str | None:
    """Clean a single extracted line; return None if it should be dropped."""
    s = raw.strip().strip("\uf0a2\uf0c0").strip()   # drop private-use box glyphs
    if not s:
        return None
    if re.fullmatch(r"[._]+", s):            # dotted/underscore answer lines (old papers)
        return None
    if re.fullmatch(r"\d+", s):              # page number
        return None
    if re.search(r"P\d{5}[A-Z]+", s):             # Pearson log code / watermark
        return None
    if s in SKIP_EXACT:
        return None
    if re.fullmatch(r"[A-Za-z]", s):         # stray vertical margin letters (D O ...)
        return None
    if re.fullmatch(r"[\u25aa\uf0b7\u2022\u25cf\u2192\u2610\uf0a2\uf0c0\s]+", s):
        return None                          # decorative bullet/box lines
    return s


def clean_lines(text: str) -> list[str]:
    """Strip boilerplate/watermark lines from a page's extracted text."""
    return [s for raw in text.splitlines() if (s := clean_one_line(raw)) is not None]


# ---------------------------------------------------------------------------
# QP parsing
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(r"^SECTION\s+([A-C])\s*$")
READ_RE = re.compile(
    r"^Read\s+Extracts?\s+(.+?)\s+in the Source Booklet before answering Question\s+(\d+)",
    re.IGNORECASE,
)
PART_A_RE = re.compile(r"^(\d+)\s*\(([a-z])\)\s*(.*)$")
PART_B_RE = re.compile(r"^\(([a-z])\)\s*(.*)$")
WHOLE_RE = re.compile(r"^(\d+)\s+(.*)$")
MARKS_RE = re.compile(r"^\((\d+)\)\s*$")
TOTAL_PAPER_RE = re.compile(r"TOTAL FOR PAPER\s*=\s*(\d+)")
UNIT_TITLE_RE = re.compile(r"unit\s*([1-4])\s*:\s*(.+)", re.IGNORECASE)
EXTRACT_HEADER_RE = re.compile(r"^Extract\s+([A-Z])\s*$")
SOURCE_SECTION_RE = re.compile(r"^Sources? for use with Section\s+([A-C])$", re.IGNORECASE)


def parse_extract_letters(s: str) -> list[str]:
    """'A, B, C, D and E' -> ['A','B','C','D','E']  /  'F and G' -> ['F','G']."""
    s = re.sub(r"\s+and\s+", ",", s, flags=re.IGNORECASE)
    return re.findall(r"[A-Z]", s)


def cited_extracts(text: str) -> list[str]:
    """Extract letters named in a stem, e.g. 'Using Extract B', 'Extracts F and G'."""
    letters = []
    for m in re.finditer(r"Extracts?\s+([A-Z](?:\s*,\s*[A-Z])*\s*(?:and\s+)?[A-Z]?)", text):
        letters += re.findall(r"[A-Z]", m.group(1))
    return sorted(set(letters))


def parse_qp(qp_path: Path) -> dict:
    """Parse the question paper into metadata, sections, questions, and extracts."""
    with pdfplumber.open(qp_path) as pdf:
        npages = len(pdf.pages)
        qpages = [(i, clean_lines(pdf.pages[i].extract_text() or "")) for i in range(npages)]

        # ---- cover metadata ---------------------------------------------------
        unit_title = None
        for line in qpages[0][1]:
            m = UNIT_TITLE_RE.match(line)
            if m:
                unit_title = m.group(2).strip()

        total_marks = None
        for _pg, lines in qpages:
            for line in lines:
                m = TOTAL_PAPER_RE.search(line)
                if m:
                    total_marks = int(m.group(1))

        # ---- classify pages: question pages vs source-booklet pages -----------
        # (Layout varies: 2019-2021 & 2023+ put the booklet at the END; 2022 puts it
        #  at the FRONT. Source pages are identified by content; every other page
        #  is a question page.)
        source_page_idxs = set(
            pg for pg, lines in qpages
            if any(SOURCE_SECTION_RE.match(l) or EXTRACT_HEADER_RE.match(l)
                   or l == "Acknowledgements" for l in lines)
        )
        question_page_idxs = [pg for pg, _lines in qpages if pg not in source_page_idxs]

        sections, questions = _parse_questions(pdf, question_page_idxs)
        extracts = _parse_extracts(pdf, source_page_idxs)

        # ---- source acknowledgements (for source_url) -------------------------
        source_map = {}
        src_pages = [p for p in qpages if p[0] in source_page_idxs]
        for _pg, lines in src_pages:
            j = 0
            while j < len(lines):
                am = re.match(r"^Extract\s+([A-Z])\s+adapted from:\s*(.*)$", lines[j])
                if am:
                    label, url = am.group(1), am.group(2).strip()
                    j += 1
                    while j < len(lines) and not re.match(r"^Extract\s+[A-Z]\s+adapted from", lines[j]):
                        url += " " + lines[j].strip()
                        j += 1
                    source_map[label] = url.strip()
                else:
                    j += 1
        for ex in extracts:
            ex["source_url"] = source_map.get(ex["label"].split()[-1])

    return {
        "unit_title": unit_title,
        "total_marks": total_marks,
        "sections": sections,
        "questions": questions,
        "extracts": extracts,
    }


def _parse_questions(pdf, question_page_idxs):
    """Return (sections, questions) from the question pages."""
    sections = {}
    questions = []
    current_section = None
    current_group = None

    # Build cleaned text lines WITH y-positions, and collect inline tables.
    lines_flat = []   # (line, pg, top, bottom)
    tables = []       # (pg, top, bottom, rows)
    for pi in question_page_idxs:
        page = pdf.pages[pi]
        page_tables = page.find_tables()
        bboxes = [t.bbox for t in page_tables]
        for tl in page.extract_text_lines():
            top = tl["top"]
            if any(b[1] <= top <= b[3] for b in bboxes):
                continue  # inside a table -> represented by the table object below
            s = clean_one_line(tl["text"])
            if s is not None:
                lines_flat.append((s, pi, top, tl["bottom"]))
        for t in page_tables:
            rows = normalize_table(t.extract())
            if rows and _is_meaningful_table(rows):
                tables.append((pi, t.bbox[1], t.bbox[3], rows))

    # begin at the first SECTION header (skips the cover page boilerplate)
    start = next((idx for idx in range(len(lines_flat)) if SECTION_RE.match(lines_flat[idx][0])), 0)

    i = start
    n = len(lines_flat)
    pending = []  # scenario/preamble lines immediately before a question

    while i < n:
        line, pg, top, bottom = lines_flat[i]

        m = SECTION_RE.match(line)
        if m:
            current_section = m.group(1)
            pending = []
            if i + 1 < n:
                rm = READ_RE.match(lines_flat[i + 1][0])
                if rm:
                    current_group = int(rm.group(2))
                    sections[current_section] = {"group": current_group,
                                                 "extracts": parse_extract_letters(rm.group(1))}
                    i += 2
                    continue
            i += 1
            continue

        # "1 (a) ..." (explicit group + part)
        m = PART_A_RE.match(line)
        if m:
            group, part, rest = int(m.group(1)), m.group(2), m.group(3).strip()
            stem, marks, i, marks_pos = _collect_stem(lines_flat, i + 1, [rest])
            questions.append(_mk_question(current_section, group, part, stem, pending, marks, pg, marks_pos))
            pending = []
            continue

        # "(b) ..." (continuation part, group inherited)
        m = PART_B_RE.match(line)
        if m and current_group is not None:
            part, rest = m.group(1), m.group(2).strip()
            stem, marks, i, marks_pos = _collect_stem(lines_flat, i + 1, [rest])
            questions.append(_mk_question(current_section, current_group, part, stem, pending, marks, pg, marks_pos))
            pending = []
            continue

        # "2 ..." (whole question, no sub-part) — must match the section's expected group
        # number, so inline data lines like "2016 sales were..." are never misread.
        m = WHOLE_RE.match(line)
        if m and current_group is not None and int(m.group(1)) == current_group:
            group, rest = current_group, m.group(2).strip()
            stem, marks, i, marks_pos = _collect_stem(lines_flat, i + 1, [rest])
            questions.append(_mk_question(current_section, group, None, stem, pending, marks, pg, marks_pos))
            pending = []
            continue

        if _is_skip_line(line):
            i += 1
            continue

        pending.append(line)  # candidate preamble
        i += 1

    _attach_tables(questions, tables)
    return sections, questions


def _is_skip_line(line: str) -> bool:
    if re.match(r"^\(?Total for Question", line):
        return True
    if re.match(r"^TOTAL FOR", line):
        return True
    return line in ("Write your answer in the space provided.",
                    "Write your answers in the spaces provided.")


def _is_meaningful_table(rows):
    """A real data table has at least one cell with letters/digits; decorative
    answer-space boxes (private-use glyphs) do not."""
    for row in rows:
        for cell in row:
            s = (cell or "").strip()
            if re.search(r"[A-Za-z0-9£€$%]", s):
                return True
    return False


def _attach_tables(questions, tables):
    """Attach each inline table to the question whose marks line is directly above it."""
    qs = [q for q in questions if q.get("_marks_pos")]
    qs.sort(key=lambda q: q["_marks_pos"])
    for pg, top, bottom, rows in tables:
        chosen = None
        for q in qs:
            mp = q["_marks_pos"]
            if (mp[0], mp[1]) < (pg, top):
                chosen = q
            else:
                break
        if chosen:
            chosen.setdefault("question_data", []).append({"type": "table", "rows": rows})


def _collect_stem(lines_flat, i, stem: list[str]):
    """Accumulate wrapped stem lines until the '(N)' marks line or the next question."""
    n = len(lines_flat)
    marks = None
    marks_pos = None
    while i < n:
        line = lines_flat[i][0]
        mm = MARKS_RE.match(line)
        if mm:
            marks = int(mm.group(1))
            marks_pos = (lines_flat[i][1], lines_flat[i][2])  # (pg, top)
            i += 1
            break
        if SECTION_RE.match(line) or PART_A_RE.match(line) or PART_B_RE.match(line):
            break
        stem.append(line)
        i += 1
    return " ".join(s.strip() for s in stem if s.strip()), marks, i, marks_pos


def _mk_question(section, group, part, stem, pending, marks, page_idx, marks_pos):
    number = f"{group}({part})" if part else str(group)
    return {
        "section": section,
        "question_group": group,
        "part": part,
        "question_number": number,
        "question_text": stem,
        "preamble_text": "\n".join(pending) if pending else None,
        "marks": marks,
        "page_ref": page_idx + 1,
        "_marks_pos": marks_pos,  # (pg, top) — used only for table association
    }


def _parse_extracts(pdf, source_page_idxs: set[int]) -> list[dict]:
    """Parse the Source Booklet into extracts, preserving prose order + table structure."""
    extracts = []
    current = None
    current_section = None
    acknowledged = False

    for pi in sorted(source_page_idxs):
        page = pdf.pages[pi]

        # ordered items: (top, kind, payload) — text lines and tables interleaved by y
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables]
        items = []
        for tl in page.extract_text_lines():
            top = tl["top"]
            if any(b[1] <= top <= b[3] for b in table_bboxes):
                continue  # inside a table -> represented by the table block below
            items.append((top, "text", tl["text"]))
        for t in tables:
            items.append((t.bbox[1], "table", normalize_table(t.extract())))
        items.sort(key=lambda x: x[0])

        for top, kind, payload in items:
            if acknowledged:
                break

            if kind == "table":
                if current is not None:
                    current["blocks"].append({"type": "table", "rows": payload})
                continue

            if payload.strip() == "Acknowledgements":
                acknowledged = True  # the rest of this page is source attributions
                break

            s = clean_one_line(payload)
            if s is None:
                continue

            sm = SOURCE_SECTION_RE.match(s)
            if sm:
                current_section = sm.group(1)
                continue

            hm = EXTRACT_HEADER_RE.match(s)
            if hm:
                if current:
                    _finalize_extract(current, extracts)
                current = {"label": f"Extract {hm.group(1)}", "section": current_section,
                           "page_ref": pi + 1, "blocks": []}
                continue

            if current is not None:
                current["blocks"].append({"type": "text", "text": s})

        if acknowledged:
            break

    if current:
        _finalize_extract(current, extracts)
    return extracts


def normalize_table(rows: list[list[str]]) -> list[list[str]]:
    """Expand cells that pdfplumber merged with newlines into separate rows."""
    out = []
    for row in rows:
        split = [(c or "").split("\n") for c in row]
        n = max(len(c) for c in split) if split else 1
        for k in range(n):
            out.append([c[k].strip() if k < len(c) else "" for c in split])
    return out


def blocks_to_text(blocks) -> str:
    lines = []
    for b in blocks:
        if b["type"] == "text":
            lines.append(b["text"])
        else:
            for row in b["rows"]:
                lines.append(" | ".join(cell for cell in row))
    return "\n".join(lines)


def _finalize_extract(current, extracts):
    blocks = current["blocks"]
    title = None
    body_blocks = []
    for b in blocks:
        if b["type"] == "text" and title is None:
            title = b["text"]  # first text line is the extract's headline
            continue
        body_blocks.append(b)
    extracts.append({
        "label": current["label"],
        "section": current["section"],
        "page_ref": current["page_ref"],
        "title": title,
        "body_text": blocks_to_text(body_blocks),
        "body_blocks": json.dumps(body_blocks, ensure_ascii=False) if body_blocks else None,
    })


# ---------------------------------------------------------------------------
# MS parsing
# ---------------------------------------------------------------------------

def clean_ms_line(s: str) -> str | None:
    s = s.strip().strip("\uf0a2\uf0c0").strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return None
    if re.search(r"P\d{5}[A-Z]+", s):
        return None
    if s == "Turn over":
        return None
    return s


def parse_ms(ms_path: Path) -> dict:
    """Parse the mark scheme into {question_number: {text, page_ref}}.

    Handles both layouts:
      * NEW (2022+): "Question <stem>..." then an "Answer" line, then "1(a) Knowledge ..."
      * OLD (2019-2021): "Question Answer Mark" / "Question Indicative content" header
        with no stem and no separate "Answer" line.
    """
    with pdfplumber.open(ms_path) as pdf:
        stream = []  # (line, page_idx, top, bottom)
        for pi in range(len(pdf.pages)):
            for tl in pdf.pages[pi].extract_text_lines():
                stream.append((tl["text"].strip(), pi, tl["top"], tl["bottom"]))

    # skip boilerplate: start at the first "Question" header
    start = next((idx for idx in range(len(stream)) if re.match(r"^Quest", stream[idx][0])), None)
    if start is None:
        return {}

    blocks = []
    cur = []
    for line, pi, top, bottom in stream[start:]:
        if re.match(r"^Quest", line) and cur:
            blocks.append(cur)
            cur = [(line, pi, top, bottom)]
        else:
            cur.append((line, pi, top, bottom))
    if cur:
        blocks.append(cur)

    result = {}
    for blk in blocks:
        qnum, qnum_page, stem, body = _parse_ms_block(blk)
        if qnum is None:
            continue
        result[qnum] = {"text": body, "page_ref": qnum_page + 1, "stem": stem,
                        "bbox": _block_bbox(blk)}
    return result


QNUM_RE = re.compile(r"^([1-9]\s*(?:\([a-z]\)|[a-z])?)(?:\s+(.*))?$")


def _norm_qnum(raw: str) -> str:
    """Normalise a question-number token: '1 (a)' -> '1(a)', '1e' -> '1(e)'."""
    raw = re.sub(r"\s+", "", raw)
    m = re.match(r"^(\d+)([a-z])$", raw)
    if m:
        return f"{m.group(1)}({m.group(2)})"
    return raw


def _is_old_block(header: str) -> bool:
    """OLD headers have no stem: 'Question Answer Mark' / 'Question Indicative content'."""
    rest = re.sub(r"^Quest\w*\s*", "", header)
    return bool(re.match(r"^(Answer\s+Mark|Indicative content)\b", rest))


_QNUM_REST_OK = ("Knowledge", "Indicative content", "QS", "Quantitative",
                 "Application", "Analysis", "Evaluation")


def _find_qnum(blk, start):
    """Find the question-number line; returns (qnum, page, index) or (None, None, None)."""
    for j in range(start, len(blk)):
        m = QNUM_RE.match(blk[j][0])
        if not m:
            continue
        num = _norm_qnum(m.group(1))
        rest = (m.group(2) or "").strip()
        if not rest:
            return num, blk[j][1], j
        if re.fullmatch(r"[.,:;\-–\s]+", rest):      # stray punctuation, e.g. "1(b) ."
            return num, blk[j][1], j
        if rest.startswith(_QNUM_REST_OK):
            return num, blk[j][1], j
    return None, None, None


def _stem_from(blk, end_exclusive):
    first = re.sub(r"^Quest\w*\s*", "", blk[0][0])
    first = re.sub(r"\s+Mark\s*$", "", first)
    parts = [first] + [item[0] for item in blk[1:end_exclusive]]
    return " ".join(p.strip() for p in parts if p.strip())


def _parse_ms_block(blk):
    if _is_old_block(blk[0][0]):
        qnum, qnum_page, qidx = _find_qnum(blk, 1)
        if qnum is None:
            return None, None, None, ""
        body = "\n".join(l for l in (clean_ms_line(item[0]) for item in blk[qidx:]) if l)
        return qnum, qnum_page, None, body

    # NEW format: stem then answer. Usually there is an "Answer" line, but a few
    # papers omit it, so fall back to scanning for the question-number line.
    ans_i = next((i for i in range(len(blk)) if re.match(r"^Answer\b", blk[i][0])), None)
    if ans_i is not None:
        qnum, qnum_page, qidx = _find_qnum(blk, ans_i + 1)
        stem_end = ans_i
    else:
        qnum, qnum_page, qidx = _find_qnum(blk, 1)
        stem_end = qidx if qidx is not None else len(blk)
    if qnum is None:
        return None, None, None, ""
    stem = _stem_from(blk, stem_end)
    body = "\n".join(l for l in (clean_ms_line(item[0]) for item in blk[qidx:]) if l)
    return qnum, qnum_page, stem, body


def _block_bbox(blk):
    """Vertical span of a block on each page, as [[page, y0, y1], ...] in PDF points.

    Used to crop the mark-scheme page image to just this question's region.
    """
    by_page = {}
    for text, pi, top, bottom in blk:
        if clean_ms_line(text) is None:  # drop page numbers / footers / Turn over
            continue
        if pi not in by_page:
            by_page[pi] = [top, bottom]
        else:
            by_page[pi][0] = min(by_page[pi][0], top)
            by_page[pi][1] = max(by_page[pi][1], bottom)
    return [[pi + 1, round(y0 - 6, 1), round(y1 + 8, 1)]
            for pi, (y0, y1) in sorted(by_page.items())]


# ---------------------------------------------------------------------------
# DB writing
# ---------------------------------------------------------------------------

def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript((ROOT / "schema.sql").read_text())
    _migrate_schema(conn)
    return conn


def _migrate_schema(conn):
    """Add columns introduced in later phases to a pre-existing database.

    CREATE TABLE IF NOT EXISTS never alters an existing table, so without this,
    re-ingesting over an older DB would fail on the new columns. ALTER TABLE
    ADD COLUMN is a no-op when the column already exists (guarded here anyway).
    """
    # questions: add columns from Phase 2 (page_ref) and Phase 3 (mark_scheme_bbox)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(questions)")}
    for col, typ in {"page_ref": "INTEGER", "mark_scheme_bbox": "TEXT",
                     "preamble_text": "TEXT", "question_data": "TEXT"}.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE questions ADD COLUMN {col} {typ}")

    # topics: if it predates topic_code, rebuild it (it is re-seeded from CSV)
    tcols = {r[1] for r in conn.execute("PRAGMA table_info(topics)")}
    if "topic_code" not in tcols:
        conn.executescript("""
            PRAGMA foreign_keys = OFF;
            DROP TABLE IF EXISTS question_topics;
            DROP TABLE IF EXISTS topics;
            CREATE TABLE topics (
                topic_id    INTEGER PRIMARY KEY,
                unit        INTEGER NOT NULL,
                category    TEXT NOT NULL DEFAULT 'syllabus',
                topic_code  TEXT NOT NULL UNIQUE,
                topic_name  TEXT NOT NULL,
                sort_order  INTEGER
            );
            CREATE TABLE question_topics (
                question_id INTEGER NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
                topic_id    INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
                PRIMARY KEY (question_id, topic_id)
            );
            PRAGMA foreign_keys = ON;
        """)
    # topics: add the Quantitative Skills category column (Phase 7)
    tcols = {r[1] for r in conn.execute("PRAGMA table_info(topics)")}
    if "category" not in tcols:
        conn.execute("ALTER TABLE topics ADD COLUMN category TEXT NOT NULL DEFAULT 'syllabus'")
    # topics: add the per-sub-topic release flag (Phase 8)
    tcols = {r[1] for r in conn.execute("PRAGMA table_info(topics)")}
    if "released" not in tcols:
        conn.execute("ALTER TABLE topics ADD COLUMN released INTEGER NOT NULL DEFAULT 1")
    _seed_qs_topics(conn)
    conn.commit()


QS_TOPICS = [
    ("QS1", "Calculate, use and understand ratios, averages and fractions"),
    ("QS2", "Calculate, use and understand percentages and percentage changes"),
    ("QS3", "Construct and interpret a range of standard graphical forms"),
    ("QS4", "Interpret index numbers"),
    ("QS5", "Calculate cost, revenue, profit and break-even"),
    ("QS6", "Calculate investment appraisal outcomes and interpret results"),
    ("QS7", "Interpret values of price and income elasticity of demand"),
    ("QS8", "Use and interpret quantitative and non-quantitative information in order to make decisions"),
    ("QS9", "Interpret, apply and analyse information in written, graphical and numerical forms"),
]


def _seed_qs_topics(conn):
    """Insert the Quantitative Skills (QS1-QS9) taxonomy rows if missing.

    These live under a separate 'quantitative' category (unit 0) so they never
    show up in the normal Unit 1-4 topic filters.
    """
    for i, (code, name) in enumerate(QS_TOPICS, 1):
        conn.execute(
            "INSERT OR IGNORE INTO topics (unit, category, topic_code, topic_name, sort_order) "
            "VALUES (0, 'quantitative', ?, ?, ?)",
            (code, name, i),
        )


def ingest(conn: sqlite3.Connection, meta: dict, qp: dict, ms: dict):
    cur = conn.cursor()

    # idempotent: clear this paper (children cascade via ON DELETE CASCADE)
    cur.execute("DELETE FROM papers WHERE paper_id = ?", (meta["paper_id"],))

    cur.execute(
        """INSERT INTO papers
           (paper_id, unit, unit_title, exam_year, exam_month, session_label,
            paper_code, paper_pdf_path, mark_scheme_pdf_path, total_marks)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (meta["paper_id"], meta["unit"], qp.get("unit_title"), meta["exam_year"],
         meta["exam_month"], meta["session_label"], meta["paper_code"],
         meta["paper_pdf_path"], meta["mark_scheme_pdf_path"], qp.get("total_marks")),
    )

    # case studies (extracts)
    cs_id_by_label = {}
    for ex in qp["extracts"]:
        cur.execute(
            """INSERT INTO case_studies
               (paper_id, label, title, section, body_text, body_blocks, source_url, page_ref)
               VALUES (?,?,?,?,?,?,?,?)""",
            (meta["paper_id"], ex["label"], ex["title"], ex["section"],
             ex["body_text"], ex.get("body_blocks"), ex.get("source_url"), ex.get("page_ref")),
        )
        cs_id_by_label[ex["label"]] = cur.lastrowid

    # questions + extract links
    for q in qp["questions"]:
        ms_block = ms.get(q["question_number"], {})
        cur.execute(
            """INSERT INTO questions
               (paper_id, unit, section, question_group, part, question_number,
                question_text, preamble_text, marks, page_ref,
                mark_scheme_text, mark_scheme_page_ref, mark_scheme_bbox, question_data)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (meta["paper_id"], meta["unit"], q["section"], q["question_group"],
             q["part"], q["question_number"], q["question_text"], q.get("preamble_text"),
             q["marks"], q.get("page_ref"),
             ms_block.get("text"),
             ms_block.get("page_ref"),
             json.dumps(ms_block["bbox"]) if ms_block.get("bbox") else None,
             json.dumps(q["question_data"]) if q.get("question_data") else None),
        )
        qid = cur.lastrowid

        # section-level extracts (every part inherits the whole section's list)
        sec = qp["sections"].get(q["section"])
        if sec:
            for letter in sec["extracts"]:
                label = f"Extract {letter}"
                if label in cs_id_by_label:
                    cur.execute(
                        "INSERT OR IGNORE INTO question_extracts (question_id, case_study_id, scope) VALUES (?,?,?)",
                        (qid, cs_id_by_label[label], "section"),
                    )

        # cited extracts (named directly in the stem)
        for letter in cited_extracts(q["question_text"]):
            label = f"Extract {letter}"
            if label in cs_id_by_label:
                cur.execute(
                    "INSERT OR IGNORE INTO question_extracts (question_id, case_study_id, scope) VALUES (?,?,?)",
                    (qid, cs_id_by_label[label], "cited"),
                )

    conn.commit()


# ---------------------------------------------------------------------------

def ingest_one(conn: sqlite3.Connection, qp_path: Path, ms_path: Path) -> dict:
    """Ingest a single paper; return a small summary dict."""
    meta = paper_meta(qp_path)
    qp = parse_qp(qp_path)
    ms = parse_ms(ms_path)
    try:
        ingest(conn, meta, qp, ms)
    except Exception:
        conn.rollback()  # never leave an uncommitted transaction behind
        raise
    missing = [q["question_number"] for q in qp["questions"] if q["question_number"] not in ms]
    return {
        "paper_id": meta["paper_id"],
        "session": meta["session_label"],
        "unit": meta["unit"],
        "code": meta["paper_code"],
        "extracts": len(qp["extracts"]),
        "questions": len(qp["questions"]),
        "ms_blocks": len(ms),
        "missing_ms": missing,
    }


def discover_papers(root: Path) -> list[Path]:
    """Return every -qp.pdf under the past-papers tree, sorted."""
    return sorted(root.rglob("*-qp.pdf"))


def main():
    ap = argparse.ArgumentParser(description="Ingest IAL Business papers into SQLite.")
    ap.add_argument("--paper", type=str, default=None,
                    help="Single paper: a directory, base stem, or -qp.pdf path.")
    ap.add_argument("--all", action="store_true",
                    help="Walk the whole IAL Business Past Papers tree and ingest every paper.")
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB))
    args = ap.parse_args()

    conn = init_db(Path(args.db))

    if args.all:
        qps = discover_papers(ROOT / "IAL Business Past Papers")
        print(f"Ingesting {len(qps)} papers into {args.db}\n")
        n_ok = 0
        problems = []
        for qp_path in qps:
            ms_path = qp_path.with_name(qp_path.name[:-len('-qp.pdf')] + '-ms.pdf')
            if not ms_path.exists():
                problems.append((qp_path.name, "missing MS"))
                continue
            try:
                s = ingest_one(conn, qp_path, ms_path)
                flag = f"  [MS missing: {', '.join(s['missing_ms'])}]" if s["missing_ms"] else ""
                print(f"{s['session']:<16} U{s['unit']} {s['code']:<9} "
                      f"extracts={s['extracts']:<2} questions={s['questions']:<2} ms={s['ms_blocks']:<2}{flag}")
                n_ok += 1
            except Exception as e:  # keep going on a single bad paper
                problems.append((qp_path.name, f"{type(e).__name__}: {e}"))
                print(f"  !! FAILED {qp_path.name}: {e}")
        conn.close()
        print(f"\nDone: {n_ok}/{len(qps)} papers ingested.")
        if problems:
            print("\nProblems:")
            for name, why in problems:
                print(f"  - {name}: {why}")
        return

    # single-paper mode
    target = args.paper or str(DEFAULT_PAPER)
    qp_path, ms_path = resolve_paths(target)
    if not qp_path.exists() or not ms_path.exists():
        sys.exit(f"ERROR: missing files:\n  {qp_path}\n  {ms_path}")

    print(f"Parsing QP : {qp_path}")
    print(f"Parsing MS : {ms_path}")
    s = ingest_one(conn, qp_path, ms_path)
    conn.close()

    print(f"\nPaper       : {s['paper_id']}  ({s['session']}, Unit {s['unit']}, {s['code']})")
    print(f"Extracts    : {s['extracts']}")
    print(f"Questions   : {s['questions']}")
    print(f"MS blocks   : {s['ms_blocks']}")
    if s["missing_ms"]:
        print("  !! MS missing for:", ", ".join(s["missing_ms"]))
    print(f"\nWrote to     : {args.db}")


if __name__ == "__main__":
    main()
