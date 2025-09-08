
import os, json, re
from typing import Dict, Any, List, Optional
from .utils import load_json, clean_text, load_env
from .ats_scoring import ats_score

def choose_backend():
    load_env()
    backend = os.getenv("MODEL_BACKEND", "openai").lower()
    name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return backend, name

def call_llm(prompt: str, system: Optional[str]=None) -> str:
    backend, model = choose_backend()
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        msgs = []
        if system: msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)
        return resp.choices[0].message.content.strip()
    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        sysmsg = system or "You are a helpful assistant."
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            system=sysmsg,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return msg.content[0].text.strip()
    elif backend == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        sysmsg = system or "You are a helpful assistant."
        chat_response = client.chat.complete(
            model=model,
            messages=[{"role":"system","content":sysmsg},{"role":"user","content":prompt}],
            temperature=0.2
        )
        return chat_response.choices[0].message.content.strip()
    else:
        # Fallback: return the prompt (for testing offline)
        return "[LLM not configured]\n" + prompt[:2000]

def build_prompt(resume_text: str, job_role: str, guidance_blobs: List[str], jd_text: str="") -> str:
    guidance = "\n\n".join(guidance_blobs[:3])
    template = f"""You are an expert resume reviewer.
Target Role: {job_role}
Optional Job Description (JD):
{jd_text[:2000]}

Domain Guidance (from internal knowledge base):
{guidance}

TASKS:
1) Give section-wise feedback (Summary, Experience, Education, Skills, Projects, Certifications).
2) List *missing* skills/keywords for this role (high priority).
3) Rewrite 3-5 bullets to be quantifiable and tailored to the JD. Use STAR actions when possible.
4) Flag vague or redundant language and suggest concise alternatives.
5) Suggest formatting/clarity improvements.
6) Provide a brief 3-line profile summary for the candidate tailored to the role.

Resume Text:
{resume_text[:6000]}

Return JSON with keys: feedback_by_section, missing_keywords, bullet_rewrites, language_fixes, formatting_suggestions, tailored_summary.
"""
    return template

def review_resume(resume_text: str, job_role: str, jd_text: str, guidance_blobs: List[str], required_skills: List[str]) -> Dict[str, Any]:
    prompt = build_prompt(resume_text, job_role, guidance_blobs, jd_text)
    system = f"You are a meticulous ATS-savvy resume coach for {job_role}."
    llm_json = call_llm(prompt, system=system)

    score, detail = ats_score(resume_text + "\n" + jd_text, required_skills)

    return {
        "ats": {"score": score, "detail": detail},
        "llm_feedback_raw": llm_json
    }
