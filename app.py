
import streamlit as st
import os, json
import pandas as pd
from components.resume_parser import extract_text_from_pdf
from components.jd_index import JDIndex, ART_DIR
from components.llm_review import review_resume
from components.utils import load_json, clean_text

st.set_page_config(page_title="Resume Reviewer", layout="wide")

st.title("🔎 LLM-Powered Resume Reviewer (ATS-aware)")
st.caption("Upload your resume, choose a target role, optionally paste a JD, and get tailored feedback.")

# Sidebar: load artifacts
st.sidebar.header("Artifacts & Role KB")
kb_loaded = False
try:
    meta = load_json(os.path.join(ART_DIR, "faiss_meta.json"), default=[])
    roles = sorted(set([m["job_position"] for m in meta]))
    skills_vocab = load_json(os.path.join(ART_DIR, "skills_vocab.json"), default=[])
    prompts = load_json(os.path.join(ART_DIR, "role_prompts.json"), default={})
    kb_loaded = True
    st.sidebar.success(f"Knowledge base loaded: {len(roles)} roles")
except Exception as e:
    st.sidebar.error("Artifacts missing. Run the training notebook first.")
    roles = []
    skills_vocab = []
    prompts = {}

# Inputs
colA, colB = st.columns(2)
with colA:
    job_role = st.selectbox("Target Job Role", options=roles if roles else ["(Run training first)"])
    jd_text = st.text_area("Paste Job Description (optional)", height=160, placeholder="Paste JD here...")

with colB:
    up = st.file_uploader("Upload Resume (PDF)", type=["pdf"], accept_multiple_files=False)
    resume_text = st.text_area("...or Paste Resume Text", height=200, placeholder="Paste resume here if not uploading PDF...")

if up is not None:
    tmp_path = os.path.join("/tmp", up.name)
    with open(tmp_path, "wb") as f:
        f.write(up.read())
    text, pages = extract_text_from_pdf(tmp_path)
    if text:
        resume_text = text
        st.success(f"Extracted text from PDF ({pages} pages). You can still edit below.")
    else:
        st.error("Could not extract text from PDF. Paste your resume text instead.")

st.divider()

if st.button("Review", type="primary"):
    if not kb_loaded:
        st.error("Artifacts not found. Please run the notebook to build the knowledge base.")
    elif not resume_text.strip():
        st.error("Provide resume text (upload or paste)." )
    else:
        # Prepare guidance blobs & required skills
        guidance_blobs = []
        required_skills = []
        for m in meta:
            if m.get("job_position") == job_role:
                guidance_blobs.append(m.get("text",""))
                required_skills.extend(m.get("skills", []))

        with st.spinner("Reviewing with LLM + ATS scoring..."):
            resp = review_resume(resume_text=resume_text, job_role=job_role, jd_text=jd_text, guidance_blobs=guidance_blobs, required_skills=required_skills)

        st.subheader("ATS Score")
        st.metric("Overall", f"{resp['ats']['score']:.1f}/100")
        st.json(resp["ats"]["detail"], expanded=False)

        st.subheader("LLM Feedback (Raw JSON)")
        st.caption("This is the JSON returned by the model. You can post-process to pretty panels.")
        st.code(resp["llm_feedback_raw"], language="json")

st.sidebar.markdown("---")
st.sidebar.info("**Privacy**: Files are processed in-memory. If you enable cloud LLMs via API keys, model providers may receive your text.")
