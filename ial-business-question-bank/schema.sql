-- Phase 1: SQLite schema for the Pearson IAL Business question bank.
-- Run with: sqlite3 data/qbank.db < schema.sql
-- (or the ingest script creates it automatically.)

PRAGMA foreign_keys = ON;

-- One row per past paper (question paper + its mark scheme).
CREATE TABLE IF NOT EXISTS papers (
    paper_id              TEXT PRIMARY KEY,   -- stable slug, e.g. "2023-unit3-2023-06-WBS13-01"
    unit                  INTEGER NOT NULL,   -- 1..4
    unit_title            TEXT,               -- e.g. "Business decisions and strategy"
    exam_year             INTEGER NOT NULL,   -- e.g. 2023
    exam_month            INTEGER NOT NULL,   -- 1 = Jan, 6 = Jun, 10 = Oct
    session_label         TEXT NOT NULL,      -- e.g. "June 2023"
    paper_code            TEXT NOT NULL,      -- e.g. "WBS13-01"
    paper_pdf_path        TEXT NOT NULL,      -- path relative to project root
    mark_scheme_pdf_path  TEXT NOT NULL,      -- path relative to project root
    total_marks           INTEGER             -- e.g. 80 (from the cover), nullable
);

-- One row per Extract (case study) in the Source Booklet.
CREATE TABLE IF NOT EXISTS case_studies (
    case_study_id  INTEGER PRIMARY KEY,
    paper_id       TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    label          TEXT NOT NULL,             -- e.g. "Extract A"
    title          TEXT,                      -- the extract's headline line, nullable
    section        TEXT,                      -- A / B / C (which Section uses it)
    body_text      TEXT NOT NULL,             -- full extract text (fallback / flattened view)
    body_blocks    TEXT,                      -- JSON: ordered [{"type":"text"|"table", ...}] blocks
    source_url     TEXT,                      -- acknowledgement URL, nullable
    page_ref       INTEGER                    -- 1-based page in the QP PDF, nullable
);

-- One row per question part. A "question group" (e.g. Question 1) with sub-parts
-- (a)-(e) becomes multiple rows, one per part, sharing question_group.
CREATE TABLE IF NOT EXISTS questions (
    question_id           INTEGER PRIMARY KEY,
    paper_id              TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    unit                  INTEGER NOT NULL,   -- denormalised from papers.unit for fast filtering
    section               TEXT NOT NULL,      -- A / B / C
    question_group        INTEGER NOT NULL,   -- 1, 2, 3 (top-level question number)
    part                  TEXT,               -- "a".."e", or NULL for a whole question (20-mark Qs)
    question_number       TEXT NOT NULL,      -- display label: "1(a)", "2", "3"
    question_text         TEXT NOT NULL,      -- the question stem
    preamble_text        TEXT,                -- scenario/data text shown above the question, nullable
    marks                 INTEGER,            -- total marks for this part
    page_ref              INTEGER,            -- 1-based page in the QP PDF where the question starts
    mark_scheme_text      TEXT,               -- full mark-scheme block for this part
    mark_scheme_page_ref  INTEGER,            -- 1-based page in the MS PDF, nullable
    mark_scheme_bbox      TEXT,               -- JSON [[page, y0, y1], ...] for cropping the MS image
    question_data         TEXT,               -- JSON [{type:'table', rows:...}] inline tables in the QP
    UNIQUE(paper_id, question_number)
);

-- Which Extracts each question depends on. A question may cite several extracts
-- and may also inherit the whole section's extract list.
CREATE TABLE IF NOT EXISTS question_extracts (
    question_id   INTEGER NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    case_study_id INTEGER NOT NULL REFERENCES case_studies(case_study_id) ON DELETE CASCADE,
    scope         TEXT NOT NULL DEFAULT 'cited',  -- 'cited' = named in the stem, 'section' = from header
    PRIMARY KEY (question_id, case_study_id, scope)
);

-- Topic taxonomy (sub-topics only). One row per syllabus sub-topic.
-- The topic_code encodes the hierarchy, e.g. '1.3.1.1' = Unit 1 · Meeting customer
-- needs · The market. We tag questions at this sub-topic level (not the lettered points).
CREATE TABLE IF NOT EXISTS topics (
    topic_id    INTEGER PRIMARY KEY,
    unit        INTEGER NOT NULL,             -- 1..4 (denormalised for fast filter); 0 for QS tags
    category    TEXT NOT NULL DEFAULT 'syllabus',  -- 'syllabus' or 'quantitative'
    topic_code  TEXT NOT NULL UNIQUE,         -- e.g. '1.3.1.1' or 'QS1'
    topic_name  TEXT NOT NULL,                -- e.g. 'The market'
    sort_order  INTEGER,                      -- display order within a unit
    released    INTEGER NOT NULL DEFAULT 1    -- 0 = held back until an admin releases it
);

-- Many-to-many between questions and topics.
CREATE TABLE IF NOT EXISTS question_topics (
    question_id INTEGER NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    topic_id    INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    PRIMARY KEY (question_id, topic_id)
);

-- App-wide key/value settings. Currently used for 'released_units': the JSON list
-- of units students are allowed to see, e.g. [1]. Written by the admin
-- "Release Units" tab; because it lives in the DB it travels with publish.sh.
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Helpful indexes for the Phase 3 filter UI.
CREATE INDEX IF NOT EXISTS idx_questions_unit    ON questions(unit);
CREATE INDEX IF NOT EXISTS idx_questions_paper   ON questions(paper_id);
CREATE INDEX IF NOT EXISTS idx_case_paper        ON case_studies(paper_id);
