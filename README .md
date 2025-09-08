# LLM-Powered Resume Reviewer (with ATS-style Scoring)

An end-to-end Python project to review resumes for a specific job role, compare them against job descriptions, provide structured, tailored feedback, and generate an improved version. Includes a Streamlit web UI, training pipeline to build a job-role knowledge base from your CSV, and an ATS-style scoring module.

---

## Features
- **Upload resume** (PDF or text) or **paste** content.
- **Target job role** selection; optionally upload/paste a **job description** (JD).
- **LLM-powered review**: missing skills/keywords, structure, clarity, quantification, tailoring tips, tone suggestions.
- **ATS-style score** (keyword match, section completeness, formatting checks, readability, achievements density).
- **Compare resume vs JD**; **highlight strengths & gaps**.
- **Export improved resume** to PDF.
- **Track multiple uploads** (session-based; extendable to a DB).
- **Training pipeline**: builds embeddings index & vocab from your CSV.
- **Models**: OpenAI/Anthropic/Mistral or local (Sentence-Transformers embeddings + prompt templates).

---

## Project Structure
```
resume_reviewer/
├─ app/
│  ├─ app.py                # Streamlit UI
│  ├─ components/
│  │  ├─ resume_parser.py   # PDF/text extraction
│  │  ├─ ats_scoring.py     # ATS-style scoring
│  │  ├─ jd_index.py        # Knowledge base: embeddings + vocab
│  │  ├─ llm_review.py      # LLM prompts & orchestration
│  │  └─ utils.py
├─ artifacts/               # Saved FAISS index, vocab, models
├─ data/
│  └─ sample_jobs.csv       # (placeholder) Your CSV goes here
├─ notebooks/
│  └─ training_pipeline.ipynb
├─ requirements.txt
└─ README.md
```

---

## Dataset Format
Your CSV must be UTF-8 with at least these columns:
- `job_position`
- `relevant_skills` (comma or pipe-separated)
- `required_qualifications`
- `job_responsibilities`
- `ideal_candidate_summary`

Place your CSV at `./data/jobs.csv` (or update paths accordingly).

---

## Quick Start

1) **Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Prepare data**
- Put your CSV (UTF-8) at `data/jobs.csv`.

3) **Run the training notebook**
- Open `notebooks/training_pipeline.ipynb` and run all cells.
- This creates artifacts in `artifacts/`:
  - `faiss_index.bin`, `faiss_meta.json`: Embedding index of roles/JDs.
  - `tfidf_job_match.pkl`: TF-IDF matcher to predict closest job role from a resume.
  - `skills_vocab.json`: Unified skills vocabulary from dataset.
  - `role_prompts.json`: Per-role prompt skeletons for LLM.

4) **Set API keys (optional)**
- Create `.env` in project root:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
MODEL_BACKEND=openai   # or anthropic | mistral
MODEL_NAME=gpt-4o-mini # example
```

5) **Launch the web app**
```bash
streamlit run app/app.py
```

6) **Use the tool**
- Upload PDF or paste resume text.
- Select job role and optionally paste JD.
- Click **Review** to get feedback, ATS score, and an improved version.
- Export as PDF if desired.

---

## Notes on Privacy
- The app does not upload your files anywhere by default.
- By default, content is processed in-memory. If you enable cloud LLMs, your content may be sent to the model provider per their terms.

---

## Optional: REST API
You can wrap core logic with FastAPI. Use `llm_review.py` functions inside an endpoint.

---

## License
MIT
