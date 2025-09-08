
import re, os, json, pathlib
from typing import List, Dict

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\u0000", " ")
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def split_csv_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[|,;\/]+", s)
    return [p.strip() for p in parts if p.strip()]

def load_env():
    from dotenv import load_dotenv
    load_dotenv()

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
