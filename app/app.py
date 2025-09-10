# app.py (with response length cap notice)
import sys
from pathlib import Path
import os
import json
import streamlit as st

# -----------------------------
# Project paths & sys.path fixes
# -----------------------------
ROOT = Path(__file__).parent.resolve()
POSSIBLE_COMPONENT_DIRS = [
    ROOT / "components",
    ROOT / "app" / "components",
    ROOT.parent / "app" / "components",
    ROOT.parent / "components",
]
for p in POSSIBLE_COMPONENT_DIRS:
    if p.exists() and p.is_dir():
        sys.path.insert(0, str(p.parent))
        sys.path.insert(0, str(p))

# -----------------------------
# ARTIFACT location discovery
# -----------------------------
try:
    from components.jd_index import ART_DIR as COMPONENTS_ART_DIR  # type: ignore
    ART_DIR = Path(COMPONENTS_ART_DIR)
except Exception:
    ART_DIR = (ROOT / "artifacts")
ART_DIR = ART_DIR.resolve()
ART_DIR.mkdir(parents=True, exist_ok=True)

def load_json_safe(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except Exception:
        return default

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🔎 Resume Reviewer (safe start)", layout="wide")
st.title("🔎 LLM-Powered Resume Reviewer (safe startup)")
st.caption("This app lazy-loads heavy components to avoid startup crashes. Upload a resume or paste text and review.")

# Sidebar: show artifact info
st.sidebar.header("Artifacts & Role KB")
meta = load_json_safe(os.path.join(ART_DIR, "faiss_meta.json"), default=[])
roles = sorted({m.get("job_position", "Unknown") for m in meta}) if meta else []
if roles:
    st.sidebar.success(f"Knowledge base loaded: {len(roles)} roles")
else:
    st.sidebar.warning("Knowledge base empty or missing. Run training notebook to generate artifacts/ first.")

# -----------------------------
# Show backend mode banner
# -----------------------------
from components.llm_review import choose_backend  # import only the light function
backend, model_name = choose_backend()
if backend == "dummy":
    st.sidebar.warning("⚠️ Running in **DUMMY MODE** — no real LLM calls, mock JSON returned.")
else:
    st.sidebar.info(f"Backend: **{backend}** | Model: **{model_name}**")

# 🔒 Always show token cap notice
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ LLM response length capped at **800 tokens** to prevent crashes.")

# -----------------------------
# Inputs
# -----------------------------
colA, colB = st.columns(2)
with colA:
    job_role = st.selectbox("Target Job Role", options=roles if roles else ["(Run training first)"])
    jd_text = st.text_area("Paste Job Description (optional)", height=160, placeholder="Paste JD here...")

with colB:
    up = st.file_uploader("Upload Resume (PDF)", type=["pdf"], accept_multiple_files=False)
    resume_text = st.text_area("...or Paste Resume Text", height=220, placeholder="Paste resume here if not uploading PDF...")

# PDF handling
if up is not None:
    tmp_path = os.path.join("/tmp", up.name)
    with open(tmp_path, "wb") as f:
        f.write(up.read())
    try:
        from components.resume_parser import extract_text_from_pdf  # type: ignore
    except Exception as e:
        st.error("Could not import resume extractor. Check components/resume_parser.py and PyMuPDF installation.")
        st.exception(e)
    else:
        text, pages = extract_text_from_pdf(tmp_path)
        if text:
            resume_text = text
            st.success(f"Extracted text from PDF ({pages} pages). You can still edit below.")
        else:
            st.error("Could not extract text from PDF. Paste your resume text instead.")

st.divider()

# -----------------------------
# Review Button
# -----------------------------
if st.button("Review", type="primary"):
    if not roles:
        st.error("Artifacts not found. Please run the training notebook first (creates artifacts/faiss_meta.json etc.).")
    elif not (resume_text and resume_text.strip()):
        st.error("Provide resume text (upload or paste).")
    else:
        guidance_blobs, required_skills = [], []
        for m in meta:
            if m.get("job_position") == job_role:
                guidance_blobs.append(m.get("text", ""))
                if isinstance(m.get("skills"), list):
                    required_skills.extend(m.get("skills"))

        with st.spinner("Running review (LLM + ATS)..."):
            try:
                from components.llm_review import review_resume  # type: ignore
                resp = review_resume(
                    resume_text=resume_text,
                    job_role=job_role,
                    jd_text=jd_text,
                    guidance_blobs=guidance_blobs,
                    required_skills=required_skills,
                )
            except Exception as e:
                st.error("Error while running review. If this calls a cloud LLM, ensure your API key is set in the environment.")
                st.exception(e)
            else:
                ats = resp.get("ats", {})
                st.subheader("ATS Score")
                if ats:
                    score = ats.get("score", 0.0)
                    st.metric("Overall", f"{score:.1f}/100")
                    st.json(ats.get("detail", {}), expanded=False)
                else:
                    st.info("No ATS scoring returned.")

                st.subheader("LLM Feedback (raw)")
                st.code(resp.get("llm_feedback_raw", "[No LLM output]"), language="json")

st.sidebar.markdown("---")
st.sidebar.info("Tip: if LLM calls fail, set `MODEL_BACKEND` or API keys in your environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).")
