"""
IAL Business question bank — Streamlit app.

Student view : browse/filter questions (case study + mark scheme as page images)
               and download papers.
Admin view   : additionally tag questions, add papers, and delete papers.

Run from the project root:
    .venv/bin/streamlit run app.py
"""

import json
import os
import sqlite3
from pathlib import Path

import streamlit as st

import ingest  # reuse the Phase 1 parse + write pipeline

try:
    import pymupdf  # PyMuPDF — renders PDF pages/regions to PNG
except ImportError:
    pymupdf = None

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "qbank.db"
PAGE_SIZE = 50  # questions per "page"; "Load more" reveals the next batch

MONTHS = {"January": 1, "June": 6, "October": 10}

UNIT_NAMES = {
    1: "Unit 1: Marketing and people",
    2: "Unit 2: Managing business activities",
    3: "Unit 3: Business decisions and strategy",
    4: "Unit 4: Global business",
}


def unit_label(u):
    return UNIT_NAMES.get(u, f"Unit {u}")


st.set_page_config(page_title="IAL Business Question Bank", layout="wide")

if pymupdf is None:
    st.error(
        "**PyMuPDF is not installed in this Python environment**, so the page images "
        "cannot be rendered. Launch the app with the project venv instead:\n\n"
        "`.venv/bin/streamlit run app.py`\n\n"
        "(or install it with `pip install pymupdf`.)"
    )


# ---------------------------------------------------------------------------
# data + rendering helpers
# ---------------------------------------------------------------------------

def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = _open_db()
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # concurrent readers + writer, no lock clash
    except Exception:
        pass
    return conn


def resolve(rel_path: str) -> Path:
    return ROOT / rel_path


def admin_password() -> str:
    return os.environ.get("ADMIN_PASSWORD") or st.secrets.get("admin_password", None) or "admin"


def admin_username() -> str:
    return os.environ.get("ADMIN_USERNAME") or st.secrets.get("admin_username", None) or "admin"


@st.cache_data(show_spinner=False, max_entries=400)
def pdf_page_png(rel_path: str, page: int) -> bytes | None:
    """Render one full PDF page to PNG (for case-study source pages)."""
    if pymupdf is None or not page:
        return None
    path = resolve(rel_path)
    if not path.exists():
        return None
    try:
        doc = pymupdf.open(path)
        if page < 1 or page > doc.page_count:
            doc.close()
            return None
        pix = doc[page - 1].get_pixmap(dpi=130)
        data = pix.tobytes("png")
        doc.close()
        return data
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=400)
def pdf_region_png(rel_path: str, page: int, y0: float, y1: float) -> bytes | None:
    """Render a vertical slice of a PDF page to PNG (for one question's mark scheme)."""
    if pymupdf is None or not page:
        return None
    path = resolve(rel_path)
    if not path.exists():
        return None
    try:
        doc = pymupdf.open(path)
        if page < 1 or page > doc.page_count:
            doc.close()
            return None
        pg = doc[page - 1]
        clip = pymupdf.Rect(0, max(0, y0), pg.rect.width, min(pg.rect.height, y1))
        pix = pg.get_pixmap(dpi=130, clip=clip)
        data = pix.tobytes("png")
        doc.close()
        return data
    except Exception:
        return None


def paper_stem(unit: int, year: int, month_name: str) -> str:
    mm = f"{MONTHS[month_name]:02d}"
    return f"{year}-unit{unit}-{year}-{mm}-WBS1{unit}-01"


def add_paper(unit: int, year: int, month_name: str, qp_bytes: bytes, ms_bytes: bytes):
    """Save an uploaded paper + mark scheme, ingest them, and preserve existing tags."""
    stem = paper_stem(unit, year, month_name)
    session = f"{month_name} {year}"
    folder = ROOT / "IAL Business Past Papers" / session / f"Unit {unit}"
    folder.mkdir(parents=True, exist_ok=True)
    qp_path = folder / f"{stem}-qp.pdf"
    ms_path = folder / f"{stem}-ms.pdf"
    qp_path.write_bytes(qp_bytes)
    ms_path.write_bytes(ms_bytes)

    wconn = _open_db()
    try:
        old_tags = {}
        qrows = wconn.execute(
            "SELECT question_id, question_number FROM questions WHERE paper_id=?", (stem,)
        ).fetchall()
        for row in qrows:
            tids = [r["topic_id"] for r in wconn.execute(
                "SELECT topic_id FROM question_topics WHERE question_id=?", (row["question_id"],)
            ).fetchall()]
            old_tags[row["question_number"]] = tids

        summary = ingest.ingest_one(wconn, qp_path, ms_path)

        for qnum, tids in old_tags.items():
            qid = wconn.execute(
                "SELECT question_id FROM questions WHERE paper_id=? AND question_number=?",
                (stem, qnum),
            ).fetchone()
            if qid:
                for tid in tids:
                    wconn.execute(
                        "INSERT OR IGNORE INTO question_topics (question_id, topic_id) VALUES (?,?)",
                        (qid["question_id"], tid),
                    )
        wconn.commit()
    finally:
        wconn.close()
    return summary, qp_path, ms_path


def delete_paper(paper_id: str):
    """Delete a paper from the DB (cascades to questions/extracts/tags) and its PDFs."""
    wconn = _open_db()
    try:
        row = wconn.execute(
            "SELECT paper_pdf_path, mark_scheme_pdf_path FROM papers WHERE paper_id=?", (paper_id,)
        ).fetchone()
        wconn.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))
        wconn.commit()
        if row:
            for rel in (row["paper_pdf_path"], row["mark_scheme_pdf_path"]):
                p = resolve(rel)
                if p.exists():
                    p.unlink()
    finally:
        wconn.close()


# ---------------------------------------------------------------------------
# login screen + access control
# ---------------------------------------------------------------------------

def render_login_screen():
    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.title("IAL Business Question Bank")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Log In As Admin", type="primary", width="stretch", key="admin_login_btn"):
            if username == admin_username() and password == admin_password():
                st.session_state["auth_state"] = "admin"
                st.session_state["view_mode"] = "Admin"
                st.rerun()
            else:
                st.error("Invalid username or password.")

        if st.button("Continue as Student", width="stretch", key="student_btn"):
            st.session_state["auth_state"] = "student"
            st.rerun()


# gate the whole app behind the login screen
if st.session_state.get("auth_state") is None:
    render_login_screen()
    st.stop()

auth_state = st.session_state.get("auth_state")  # "student" or "admin"

st.sidebar.header("Access")

if auth_state == "admin":
    st.sidebar.caption("Logged in as: Admin")
    st.session_state.setdefault("view_mode", "Admin")
    view = st.sidebar.radio("View", ["Student", "Admin"], key="view_mode")
    is_admin = (view == "Admin")
else:
    st.sidebar.caption("Logged in as: Student")
    is_admin = False

if st.sidebar.button("Log out", key="logout_btn"):
    st.session_state["auth_state"] = None
    st.session_state.pop("view_mode", None)
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Filters")

conn = get_conn()

unit = st.sidebar.selectbox(
    "Unit",
    ["All", 1, 2, 3, 4],
    format_func=lambda u: "All Units" if u == "All" else unit_label(u),
    key="unit_sel",
)

if unit == "All":
    topic_rows = conn.execute(
        "SELECT topic_id, unit, topic_code, topic_name FROM topics "
        "WHERE category='syllabus' ORDER BY topic_code"
    ).fetchall()
else:
    topic_rows = conn.execute(
        "SELECT topic_id, unit, topic_code, topic_name FROM topics "
        "WHERE category='syllabus' AND unit=? ORDER BY topic_code",
        (unit,),
    ).fetchall()

topic_ids = ["ALL"] + [r["topic_id"] for r in topic_rows] + ["QS"]
topic_label = {"ALL": "All Topics", "QS": "Quantitative Skills"}
for r in topic_rows:
    topic_label[r["topic_id"]] = f"{r['topic_code']} — {r['topic_name']}"

valid_ids = set(topic_ids)
if "topic_ms" not in st.session_state:
    st.session_state["topic_ms"] = ["ALL"]
    st.session_state["topic_ms_prev"] = ["ALL"]
elif not all(t in valid_ids for t in st.session_state["topic_ms"]):
    st.session_state["topic_ms"] = ["ALL"]
    st.session_state["topic_ms_prev"] = ["ALL"]


def _on_topics_change():
    """Keep 'All Topics' and specific topics mutually exclusive."""
    sel = st.session_state["topic_ms"]
    if not sel:
        st.session_state["topic_ms"] = ["ALL"]
    elif "ALL" in sel:
        prev = st.session_state.get("topic_ms_prev", ["ALL"])
        if "ALL" in prev:
            st.session_state["topic_ms"] = [x for x in sel if x != "ALL"]
        else:
            st.session_state["topic_ms"] = ["ALL"]
    st.session_state["topic_ms_prev"] = st.session_state["topic_ms"]


st.sidebar.multiselect(
    "Topics",
    topic_ids,
    format_func=lambda t: topic_label[t],
    key="topic_ms",
    on_change=_on_topics_change,
)

topics = st.session_state["topic_ms"]
selected_topics = [t for t in topics if t not in ("ALL", "QS")]
has_qs = "QS" in topics

# --- Quantitative Skills: only revealed when 'Quantitative Skills' is chosen --
selected_qs = []
if has_qs:
    qs_rows = conn.execute(
        "SELECT topic_id, topic_code, topic_name FROM topics "
        "WHERE category='quantitative' ORDER BY sort_order"
    ).fetchall()
    qs_ids = [r["topic_id"] for r in qs_rows]
    qs_label = {r["topic_id"]: f"{r['topic_code']} — {r['topic_name']}" for r in qs_rows}
    if "qs_ms" not in st.session_state:
        st.session_state["qs_ms"] = []
    elif not all(t in qs_ids for t in st.session_state["qs_ms"]):
        st.session_state["qs_ms"] = []
    st.sidebar.multiselect(
        "Quantitative skill",
        qs_ids,
        format_func=lambda t: qs_label[t],
        key="qs_ms",
    )
    st.sidebar.caption("Leave empty to show any question that tests a quantitative skill.")
    selected_qs = st.session_state.get("qs_ms", [])

st.sidebar.caption("Pick a Unit and one or more Topics to see matching questions.")


def build_filter():
    conds, params = [], []
    if unit != "All":
        conds.append("q.unit = ?")
        params.append(unit)
    if selected_topics:
        ph = ",".join("?" * len(selected_topics))
        conds.append(
            "EXISTS (SELECT 1 FROM question_topics qt WHERE qt.question_id = q.question_id "
            f"AND qt.topic_id IN ({ph}))"
        )
        params.extend(selected_topics)
    if has_qs:
        if selected_qs:
            ph = ",".join("?" * len(selected_qs))
            conds.append(
                "EXISTS (SELECT 1 FROM question_topics qt WHERE qt.question_id = q.question_id "
                f"AND qt.topic_id IN ({ph}))"
            )
            params.extend(selected_qs)
        else:
            conds.append(
                "EXISTS (SELECT 1 FROM question_topics qt JOIN topics t USING(topic_id) "
                "WHERE qt.question_id = q.question_id AND t.category='quantitative')"
            )
    return (" AND ".join(conds) if conds else "1=1"), params


# ---------------------------------------------------------------------------
# tabs (role-aware)
# ---------------------------------------------------------------------------

tab_labels = ["Browse Questions", "Download Papers"]
if is_admin:
    tab_labels += ["Tag Questions", "Add Paper", "Delete Paper"]

tabs = st.tabs(tab_labels)
tab_browse = tabs[0]
tab_download = tabs[1]
tab_tag = tabs[2] if is_admin else None
tab_add = tabs[3] if is_admin else None
tab_delete = tabs[4] if is_admin else None

# ===========================================================================
# Tab 1 — filtered questions
# ===========================================================================
with tab_browse:
    st.title("IAL Business Question Bank")
    syl_path = ROOT / "IAL Business Syllabus.pdf"
    if syl_path.exists():
        st.download_button(
            "Download syllabus",
            data=syl_path.read_bytes(),
            file_name=syl_path.name,
            mime="application/pdf",
            key="dl-syllabus",
        )
    else:
        st.caption("Syllabus PDF not found.")

    no_topic_selected = (not selected_topics and not has_qs)

    with st.expander("How to use this app", expanded=no_topic_selected):
        st.markdown(
            """
            **Welcome.** This question bank contains past-paper questions from the Pearson IAL
            Business Syllabus.

            1. Choose a **Unit** and one or more **Topics** in the sidebar to view questions.
               Pick **Quantitative Skills** to see questions that test quantitative skills.
            2. Each result shows the question, the **case study (Extract)** and the **mark scheme**.
            3. Open a question's *case study* or *mark scheme* expander to see the original page.

            *Select topics in the sidebar to get started.*
            """
        )

    sig = (unit, tuple(sorted(selected_topics)))
    if st.session_state.get("filt_sig") != sig:
        st.session_state["filt_sig"] = sig
        st.session_state["shown"] = PAGE_SIZE
    shown = st.session_state.setdefault("shown", PAGE_SIZE)

    if no_topic_selected:
        st.info("Select one or more Topics in the sidebar to view questions.")
        total = 0
        rows = []
    else:
        where, params = build_filter()
        base_from = "FROM questions q JOIN papers p USING(paper_id)"

        total = conn.execute(f"SELECT COUNT(*) {base_from} WHERE {where}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT q.question_id, q.unit, q.question_number, q.question_text, q.marks,
                       q.mark_scheme_bbox, q.mark_scheme_page_ref,
                       q.preamble_text, q.question_data,
                       p.session_label, p.paper_code, p.exam_year, p.exam_month,
                       p.paper_pdf_path, p.mark_scheme_pdf_path
                {base_from}
                WHERE {where}
                ORDER BY q.unit, p.exam_year, p.exam_month, q.question_group, q.part
                LIMIT {shown}""",
            params,
        ).fetchall()

        count_msg = f"**{total}** matching question{'s' if total != 1 else ''}"
        if total > shown:
            count_msg += f" · showing {shown}"
        st.markdown(count_msg)

        if not rows:
            st.info("No questions match the current filters.")

    for q in rows:
        st.markdown(
            f"### Q{q['question_number']} · {q['marks']} marks · {q['session_label']} "
            f"· {q['paper_code']}"
        )

        tags = conn.execute(
            "SELECT t.topic_code, t.topic_name, t.category FROM question_topics qt "
            "JOIN topics t USING(topic_id) WHERE qt.question_id=? ORDER BY t.topic_code",
            (q["question_id"],),
        ).fetchall()
        syl_tags = [t for t in tags if t["category"] == "syllabus"]
        qs_tags = [t for t in tags if t["category"] == "quantitative"]
        if syl_tags:
            tag_html = " &nbsp; ".join(
                f"<code>{t['topic_code']} — {t['topic_name']}</code>" for t in syl_tags
            )
            st.markdown(f"**Syllabus:** {tag_html}", unsafe_allow_html=True)
        if qs_tags:
            qs_html = " &nbsp; ".join(
                f"<code>{t['topic_code']} — {t['topic_name']}</code>" for t in qs_tags
            )
            st.markdown(f"**Quantitative Skills:** {qs_html}", unsafe_allow_html=True)

        if q["preamble_text"]:
            st.markdown(f"*{q['preamble_text']}*")

        st.write(q["question_text"])

        if q["question_data"]:
            try:
                for b in json.loads(q["question_data"]):
                    if b.get("type") == "table":
                        st.table(b["rows"])
            except Exception:
                pass

        with st.expander("View Case Study (Extract)", expanded=False):
            extracts = conn.execute(
                """SELECT cs.label, cs.title, cs.page_ref
                   FROM question_extracts qe JOIN case_studies cs USING(case_study_id)
                   WHERE qe.question_id=? AND qe.scope='cited' ORDER BY cs.label""",
                (q["question_id"],),
            ).fetchall()
            if not extracts:
                extracts = conn.execute(
                    """SELECT cs.label, cs.title, cs.page_ref
                       FROM question_extracts qe JOIN case_studies cs USING(case_study_id)
                       WHERE qe.question_id=? AND qe.scope='section' ORDER BY cs.label""",
                    (q["question_id"],),
                ).fetchall()

            if extracts:
                pages = sorted({cs["page_ref"] for cs in extracts if cs["page_ref"]})
                for pg in pages:
                    png = pdf_page_png(q["paper_pdf_path"], pg)
                    if png:
                        st.image(png, caption=f"Source booklet — page {pg}", width="stretch")
            else:
                st.caption("No case study linked to this question.")

        with st.expander("View Mark Scheme", expanded=False):
            bbox = json.loads(q["mark_scheme_bbox"]) if q["mark_scheme_bbox"] else None
            if bbox:
                for pg, y0, y1 in bbox:
                    png = pdf_region_png(q["mark_scheme_pdf_path"], pg, y0, y1)
                    if png:
                        st.image(png, caption=f"Mark scheme — page {pg}", width="stretch")
            elif q["mark_scheme_page_ref"]:
                png = pdf_page_png(q["mark_scheme_pdf_path"], q["mark_scheme_page_ref"])
                if png:
                    st.image(png, caption="Mark scheme", width="stretch")

        st.divider()

    if total > shown:
        if st.button(f"Load more (showing {shown} of {total})"):
            st.session_state["shown"] = shown + PAGE_SIZE
            st.rerun()

# ===========================================================================
# Tab 2 — paper downloads
# ===========================================================================
with tab_download:
    st.title("Download Papers")

    d_unit = st.selectbox("Unit", [1, 2, 3, 4], format_func=unit_label, key="dl_unit")
    d_sessions = [
        r["session_label"]
        for r in conn.execute(
            "SELECT DISTINCT session_label, MIN(exam_year) y, MIN(exam_month) m "
            "FROM papers WHERE unit=? GROUP BY session_label ORDER BY y, m",
            (d_unit,),
        )
    ]
    d_session = st.selectbox("Session", d_sessions, key="dl_session")
    d_papers = conn.execute(
        "SELECT * FROM papers WHERE unit=? AND session_label=? ORDER BY paper_code",
        (d_unit, d_session),
    ).fetchall()
    d_paper = st.selectbox("Paper", [p["paper_code"] for p in d_papers], key="dl_paper")
    paper_dl = next(p for p in d_papers if p["paper_code"] == d_paper)

    st.markdown(
        f"### {paper_dl['session_label']} · {unit_label(paper_dl['unit'])} · {paper_dl['paper_code']}"
    )
    st.caption(
        f"{paper_dl['unit_title'] or ''} · {paper_dl['total_marks']} marks · "
        f"{Path(paper_dl['paper_pdf_path']).name}"
    )

    col_qp, col_ms = st.columns(2)
    with col_qp:
        if resolve(paper_dl["paper_pdf_path"]).exists():
            st.download_button(
                "Download question paper",
                data=resolve(paper_dl["paper_pdf_path"]).read_bytes(),
                file_name=Path(paper_dl["paper_pdf_path"]).name,
                mime="application/pdf",
                key=f"dlp-qp-{paper_dl['paper_id']}",
            )
        else:
            st.error("Question paper PDF is missing on disk.")
    with col_ms:
        if resolve(paper_dl["mark_scheme_pdf_path"]).exists():
            st.download_button(
                "Download mark scheme",
                data=resolve(paper_dl["mark_scheme_pdf_path"]).read_bytes(),
                file_name=Path(paper_dl["mark_scheme_pdf_path"]).name,
                mime="application/pdf",
                key=f"dlp-ms-{paper_dl['paper_id']}",
            )
        else:
            st.warning("Mark scheme PDF is not available for this paper.")

# ---------------------------------------------------------------------------
# Admin-only tabs
# ---------------------------------------------------------------------------

if is_admin:
    # =========================================================================
    # Tab 3 — tag questions to sub-topics
    # =========================================================================
    with tab_tag:
        st.title("Tag Questions")
        st.caption(
            "Assign each question to its syllabus sub-topic(s). Tags drive the filters in "
            "the Browse tab. (You can also bulk-load a CSV via `tag.py --tags tags.csv`.)"
        )

        t_unit = st.selectbox("Unit", [1, 2, 3, 4], format_func=unit_label, key="t_unit")
        t_sessions = [
            r["session_label"]
            for r in conn.execute(
                "SELECT DISTINCT session_label, MIN(exam_year) y, MIN(exam_month) m "
                "FROM papers WHERE unit=? GROUP BY session_label ORDER BY y, m",
                (t_unit,),
            )
        ]
        t_session = st.selectbox("Session", t_sessions, key="t_session")
        t_papers = conn.execute(
            "SELECT * FROM papers WHERE unit=? AND session_label=? ORDER BY paper_code",
            (t_unit, t_session),
        ).fetchall()
        t_paper = st.selectbox("Paper", [p["paper_code"] for p in t_papers], key="t_paper")
        paper = next(p for p in t_papers if p["paper_code"] == t_paper)

        unit_topics = conn.execute(
            "SELECT topic_id, topic_code, topic_name FROM topics "
            "WHERE category='syllabus' AND unit=? ORDER BY topic_code",
            (t_unit,),
        ).fetchall()
        tid_label = {t["topic_id"]: f"{t['topic_code']} — {t['topic_name']}" for t in unit_topics}

        qs_topics = conn.execute(
            "SELECT topic_id, topic_code, topic_name FROM topics "
            "WHERE category='quantitative' ORDER BY sort_order"
        ).fetchall()
        qs_tid_label = {t["topic_id"]: f"{t['topic_code']} — {t['topic_name']}" for t in qs_topics}
        qs_tid_set = set(qs_tid_label)

        def save_tags(qid, syllabus_ids, qs_ids):
            cur = conn.cursor()
            cur.execute("DELETE FROM question_topics WHERE question_id=?", (qid,))
            for tid in list(syllabus_ids) + list(qs_ids):
                cur.execute(
                    "INSERT OR IGNORE INTO question_topics (question_id, topic_id) VALUES (?,?)",
                    (qid, tid),
                )
            conn.commit()

        questions = conn.execute(
            "SELECT question_id, question_number, question_text, preamble_text, question_data, marks "
            "FROM questions WHERE paper_id=? ORDER BY question_group, part",
            (paper["paper_id"],),
        ).fetchall()

        for q in questions:
            st.markdown(f"### Q{q['question_number']} · {q['marks']} marks")
            if q["preamble_text"]:
                st.markdown(f"*{q['preamble_text']}*")
            st.write(q["question_text"])
            if q["question_data"]:
                try:
                    for b in json.loads(q["question_data"]):
                        if b.get("type") == "table":
                            st.table(b["rows"])
                except Exception:
                    pass
            current = [
                r["topic_id"]
                for r in conn.execute(
                    "SELECT topic_id FROM question_topics WHERE question_id=?", (q["question_id"],)
                )
            ]
            cur_syl = [t for t in current if t not in qs_tid_set]
            cur_qs = [t for t in current if t in qs_tid_set]
            sel = st.multiselect(
                "Syllabus topics",
                [t["topic_id"] for t in unit_topics],
                default=cur_syl,
                format_func=lambda tid: tid_label[tid],
                key=f"tag_sel_{q['question_id']}",
            )
            sel_qs = st.multiselect(
                "Quantitative Skills",
                [t["topic_id"] for t in qs_topics],
                default=cur_qs,
                format_func=lambda tid: qs_tid_label[tid],
                key=f"tag_qs_{q['question_id']}",
            )
            if st.button("Save tags", key=f"save_{q['question_id']}"):
                save_tags(q["question_id"], sel, sel_qs)
                st.rerun()
            st.divider()

    # =========================================================================
    # Tab 4 — upload a new paper
    # =========================================================================
    with tab_add:
        st.title("Add Paper")
        st.caption(
            "Upload a new question paper and its mark scheme. The app extracts the "
            "questions and case studies and adds them to the bank (then tag them in "
            "the 'Tag Questions' tab)."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            up_unit = st.selectbox("Unit", [1, 2, 3, 4], format_func=unit_label, key="up_unit")
        with c2:
            up_year = st.number_input("Year", min_value=2019, max_value=2035, value=2027, step=1, key="up_year")
        with c3:
            up_month = st.selectbox("Session", list(MONTHS.keys()), key="up_month")

        up_qp = st.file_uploader("Question paper (PDF)", type=["pdf"], key="up_qp")
        up_ms = st.file_uploader("Mark scheme (PDF)", type=["pdf"], key="up_ms")

        stem = paper_stem(up_unit, int(up_year), up_month)
        if conn.execute("SELECT 1 FROM papers WHERE paper_id=?", (stem,)).fetchone():
            st.warning(
                f"Paper `{stem}` already exists. Uploading will overwrite it "
                "(its existing tags are preserved)."
            )

        if st.button("Add paper"):
            if up_qp is None or up_ms is None:
                st.error("Please upload both the question paper and the mark scheme.")
            else:
                try:
                    summary, qp_path, _ms_path = add_paper(
                        up_unit, int(up_year), up_month, up_qp.getvalue(), up_ms.getvalue()
                    )
                    st.success(
                        f"Added {summary['paper_id']} — {summary['questions']} questions, "
                        f"{summary['extracts']} extracts."
                    )
                    if summary.get("missing_ms"):
                        st.warning(
                            "Some questions could not be matched to a mark-scheme block: "
                            + ", ".join(summary["missing_ms"])
                        )
                    st.info(f"Saved to {qp_path}")
                    st.info("Remember to tag the new questions in the 'Tag Questions' tab.")
                except Exception as e:
                    st.error(f"Failed to add paper: {e}")

    # =========================================================================
    # Tab 5 — delete a paper
    # =========================================================================
    with tab_delete:
        st.title("Delete Paper")
        st.caption("Permanently delete a paper and its question paper / mark scheme PDFs.")

        e_unit = st.selectbox("Unit", [1, 2, 3, 4], format_func=unit_label, key="e_unit")
        e_sessions = [
            r["session_label"]
            for r in conn.execute(
                "SELECT DISTINCT session_label, MIN(exam_year) y, MIN(exam_month) m "
                "FROM papers WHERE unit=? GROUP BY session_label ORDER BY y, m",
                (e_unit,),
            )
        ]
        e_session = st.selectbox("Session", e_sessions, key="e_session")
        e_papers = conn.execute(
            "SELECT * FROM papers WHERE unit=? AND session_label=? ORDER BY paper_code",
            (e_unit, e_session),
        ).fetchall()
        e_paper = st.selectbox("Paper", [p["paper_code"] for p in e_papers], key="e_paper")
        paper_del = next(p for p in e_papers if p["paper_code"] == e_paper)

        n_questions = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE paper_id=?", (paper_del["paper_id"],)
        ).fetchone()[0]
        st.warning(
            f"This will permanently delete **{paper_del['paper_id']}** "
            f"({n_questions} questions) and its two PDF files."
        )

        confirm = st.checkbox("I understand this cannot be undone.", key="del_confirm")
        if st.button("Delete paper", disabled=not confirm, key="del_button"):
            delete_paper(paper_del["paper_id"])
            st.success(f"Deleted {paper_del['paper_id']}.")
            st.rerun()
