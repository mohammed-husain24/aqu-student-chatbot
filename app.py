
#   Al-Quds University Chatbot  
#    Optional Groq LLM (Llama 3.1 8B)
#    Multi-turn memory 
#    Clickable sources links 
#    /source/<file> fallback viewer/redirect


import os
import re
import string
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import numpy as np
from flask import Flask, request, render_template, session, jsonify, abort, redirect
from bs4 import BeautifulSoup

# ---------- Optional deps ----------
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import requests
except Exception:
    requests = None

from sentence_transformers import SentenceTransformer
load_dotenv("code.env")

# ---------- PATHS & FLASK SETUP ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# ---------- STOPWORDS & TOKENIZATION ----------
STOPWORDS = {
    "the", "a", "an", "and", "or", "not", "of", "to", "in", "for", "on", "at", "by", "with",
    "is", "are", "was", "were", "be", "as", "that", "this", "it", "its", "from", "about",
    "can", "how", "what", "when", "where", "which", "who", "whom", "why", "there", "do", "does",
    # Arabic stop-like words
    "ما", "ماذا", "كيف", "هل", "في", "عن", "من", "على", "الى", "إلى", "او", "أو", "لماذا", "متى",
}


def _keywords(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    toks = text.split()
    return {t for t in toks if t not in STOPWORDS and len(t) > 2}


def _clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r":contentReference\[[^\]]*\]", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


# ---------- MEMORY EXTRACTION ----------
def extract_memory_facts(user_text: str) -> Dict[str, str]:
    """
    Simple rule-based memory extraction for the current conversation.
    """
    t = (user_text or "").strip()
    facts: Dict[str, str] = {}

    # English: "I got 85 in Tawjihi" / "my score is 85 in tawjihi"
    m = re.search(
        r"\b(?:i\s*got|my\s*score\s*is)\s*(\d{2,3})\s*(?:in|on)\s*(tawjihi|tawjihi exam)\b",
        t,
        re.I,
    )
    if m:
        facts["tawjihi_score"] = m.group(1)

    # Arabic: "جبت 85 بالتوجيهي" / "معدلي 85 توجيهي"
    m = re.search(r"(?:جبت|معدلي|علامتي)\s*(\d{2,3})\s*(?:بالتوجيهي|في\s*التوجيهي|توجيهي)", t)
    if m:
        facts["tawjihi_score"] = m.group(1)

    # Name: "My name is Mohammad" / "اسمي محمد"
    m = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s]{1,30})\b", t, re.I)
    if m:
        facts["name"] = m.group(1).strip()

    m = re.search(r"(?:اسمي)\s+([^\s]{2,20})", t)
    if m:
        facts["name"] = m.group(1).strip()

    return facts


def memory_to_text(mem: Dict[str, str], lang_is_ar: bool) -> str:
    if not mem:
        return "(none)"
    lines: List[str] = []
    for k, v in mem.items():
        if k == "tawjihi_score":
            lines.append(("معدل التوجيهي" if lang_is_ar else "Tawjihi score") + f": {v}")
        elif k == "name":
            lines.append(("الاسم" if lang_is_ar else "Name") + f": {v}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(f"- {x}" for x in lines)


# ---------- SMARTER CHUNKING ----------
CHUNK_MAX_LEN = 900
CHUNK_MIN_LEN = 250


def _split_by_paragraphs(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text.strip()]


def _split_by_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?؟])\s+", text or "")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [(text or "").strip()]


def _recursive_chunk(text: str, max_len: int = CHUNK_MAX_LEN) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    paras = _split_by_paragraphs(text)
    if len(paras) > 1:
        out: List[str] = []
        for p in paras:
            if len(p) <= max_len:
                out.append(p)
            else:
                out.extend(_recursive_chunk(p, max_len=max_len))
        return out

    sents = _split_by_sentences(text)
    if len(sents) > 1:
        out: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for s in sents:
            s_len = len(s) + 1
            if cur and cur_len + s_len > max_len:
                out.append(" ".join(cur).strip())
                cur = [s]
                cur_len = len(s)
            else:
                cur.append(s)
                cur_len += s_len
        if cur:
            out.append(" ".join(cur).strip())

        final: List[str] = []
        for c in out:
            if len(c) <= max_len:
                final.append(c)
            else:
                start = 0
                while start < len(c):
                    final.append(c[start:start + max_len].strip())
                    start += max_len
        return final

    out = []
    start = 0
    while start < len(text):
        out.append(text[start:start + max_len].strip())
        start += max_len
    return out


# ---------- FILE LOADING ----------
def _read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return _clean_text(f.read())

        if ext in [".html", ".htm"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return _clean_text(soup.get_text(separator=" ", strip=True))

        if ext == ".pdf" and PyPDF2 is not None:
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        text.append(page.extract_text() or "")
                    except Exception:
                        pass
            return _clean_text("\n".join(text))
    except Exception:
        return ""
    return ""


def load_knowledge() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for root, _, files in os.walk(DATA_DIR):
        for fn in files:
            path = os.path.join(root, fn)
            text = _read_text_from_file(path)
            if not text:
                continue
            rel = os.path.relpath(path, DATA_DIR)
            chunks = _recursive_chunk(text, max_len=CHUNK_MAX_LEN)
            for c in chunks:
                docs.append({"source": rel, "text": _clean_text(c)})
    return docs


print(">>> DATA_DIR      =", DATA_DIR)
DOCS = load_knowledge()
print(f">>> Loaded {len(DOCS)} text chunks from data/")
if not DOCS:
    raise RuntimeError("No documents found in data/. Please check your data directory.")

# ---------- EMBEDDINGS ----------
print(">>> Loading sentence-transformers model (first run may download it)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
texts = [d["text"] for d in DOCS]
DOC_EMBS = model.encode(texts, normalize_embeddings=True)
DOC_EMBS = np.asarray(DOC_EMBS, dtype="float32")
print(">>> Embeddings shape:", DOC_EMBS.shape)

# ---------- OPTIONAL GROQ ----------
# Put your key in env variable: GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

GROQ_ENABLED = bool(GROQ_API_KEY) and (requests is not None)

print(">>> GROQ_API_KEY present:", bool(GROQ_API_KEY))
print(">>> requests available  :", requests is not None)
print(">>> Groq LLM fallback", "ENABLED" if GROQ_ENABLED else "DISABLED")


def _strip_llm_output(content: str) -> str:
    if not content:
        return content
    markers = ["final answer:", "answer:", "final_answer:", "الجواب:", "الإجابة:"]
    lower = content.lower()
    for m in markers:
        idx = lower.find(m)
        if idx != -1:
            content = content[idx + len(m):].strip()
            break
    if len(content) > 1400:
        content = content[:1400].rsplit(" ", 1)[0] + "…"
    return content.strip()


def groq_answer(query: str, context: str, memory_text: str, chat_context: str) -> Optional[str]:
    if not GROQ_ENABLED:
        return None

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        system_msg = (
            "You are a helpful assistant for Al-Quds University students. "
            "Answer ONLY using the provided university context. "
            "If the answer is not clearly in the context, say you do not know.\n\n"
            "Rules:\n"
            "- Use conversation memory ONLY for personalization (e.g., student's Tawjihi score), "
            "not for university facts.\n"
            "- Do not invent policies, fees, phone numbers, or URLs.\n"
            "- If a city like 'Dubai' is not mentioned in context, say you do not see info about it.\n\n"
            "Style:\n"
            "- Be brief and clear (2–6 bullet points max).\n"
            "- Answer in the same language as the question."
        )

        user_content = (
            f"Conversation memory:\n{memory_text}\n\n"
            f"Recent chat context:\n{chat_context}\n\n"
            f"University context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer now."
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 450,
        }

        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return _strip_llm_output(content) or None
    except Exception as e:
        print(">>> Groq error:", e)
        return None


# ---------- RETRIEVAL + SAFETY ----------
SIM_THRESHOLD = 0.35
HIGH_SIM_THRESHOLD = 0.60  # unused now, kept for future tuning

QUERY_ANSWER_CACHE: Dict[str, Dict[str, Any]] = {}


def classify_question(query: str) -> str:
    q = (query or "").lower()
    off_domain = [
        "hogwarts", "harvard", "mit", "oxford", "cambridge", "stanford",
        "yale", "princeton", "google", "facebook", "instagram", "netflix",
    ]
    if any(m in q for m in off_domain):
        return "off_domain"

    sensitive = [
        "salary", "wage", "income", "how much does the president earn",
        "how much does the dean earn", "personal phone", "personal email",
        "national id", "id number",
    ]
    if any(m in q for m in sensitive):
        return "sensitive"

    return "normal"


def retrieve_with_scores(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not (query or "").strip():
        return []
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = DOC_EMBS @ q_emb[0]
    top_idx = np.argsort(-sims)[:top_k]

    q_kw = _keywords(query)
    out: List[Dict[str, Any]] = []
    for idx in top_idx:
        doc = DOCS[idx]
        score = float(sims[idx])
        overlap = len(q_kw & _keywords(doc["text"]))
        out.append({"score": score, "overlap": overlap, "doc": doc})
    return out


def answer_query(query: str, memory_text: str, chat_context: str) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"answer": "Please type a question.", "sources": []}

    cached = QUERY_ANSWER_CACHE.get(query)
    if cached:
        return cached

    q_type = classify_question(query)
    if q_type == "off_domain":
        ans = (
            "I’m designed to answer questions about Al-Quds University only "
            "(admissions, programs, scholarships, housing, campus life, etc.). "
            "Your question seems to be about something else, so I can’t answer it."
        )
        out = {"answer": ans, "sources": []}
        QUERY_ANSWER_CACHE[query] = out
        return out

    if q_type == "sensitive":
        ans = (
            "I don’t have access to private salary details or confidential personal information "
            "about university staff or officials. For official information, please contact the "
            "university administration through official channels."
        )
        out = {"answer": ans, "sources": []}
        QUERY_ANSWER_CACHE[query] = out
        return out

    # Short follow-up handling: if the user sends a very brief utterance such as
    # "ok", "meets", "تمام", "بنفع" that is not a proper question, ask them
    # to clarify rather than generate a potentially hallucinated answer. We only
    # trigger this for known follow-up words or phrases, not for standalone
    # topic requests like "housing" or "scholarships".
    if not _is_question(query):
        stripped = (query or "").strip().lower()
        # Remove diacritics or punctuation for Arabic detection
        stripped_ar = re.sub(r"[\s\u064B-\u065F]", "", stripped)
        follow_words = {
            "ok", "okay", "thanks", "thank you", "meets", "cool", "great", "fine", "good", "yes", "no",
            "تمام", "بنفع", "ممتاز", "كويس", "نعم", "لا", "شكرا", "شكراً"
        }
        if stripped in follow_words or stripped_ar in follow_words:
            if _is_arabic(query):
                ans = "هل يمكنك توضيح سؤالك أو طلبك؟"
            else:
                ans = "Could you please clarify your question or provide more details?"
            out = {"answer": ans, "sources": []}
            QUERY_ANSWER_CACHE[query] = out
            return out

    # Retrieve top-K chunks from the knowledge base using cosine similarity.
    # Each hit includes a similarity score and keyword overlap for later filtering.
    hits = retrieve_with_scores(query, top_k=5)
    if not hits:
        ans = "I don’t know that one yet. Try rephrasing or ask about admissions, faculties, housing, or scholarships."
        out = {"answer": ans, "sources": []}
        QUERY_ANSWER_CACHE[query] = out
        return out

    best = hits[0]
    best_score = best["score"]
    best_overlap = best["overlap"]
    # Determine if the query is Arabic to adjust the threshold slightly lower
    is_ar = _is_arabic(query)
    threshold = SIM_THRESHOLD - 0.05 if is_ar else SIM_THRESHOLD

    # Only block if similarity is below the threshold AND there is no keyword overlap.
    # Confidence check: only block when both the cosine similarity and keyword overlap
    # are too low. Arabic queries get a slightly lower threshold to accommodate
    # language differences. When blocked, return a polite message instead of
    # hallucinating or dumping content.
    if (best_score < threshold) and (best_overlap <= 0):
        ans = "I don’t see specific information about that in the university sources."
        out = {"answer": ans, "sources": []}
        QUERY_ANSWER_CACHE[query] = out
        return out

    top_docs = hits[:3]
    sources = sorted(set(d["doc"]["source"] for d in top_docs))
    context_text = "\n\n".join(d["doc"]["text"] for d in top_docs)

    # Before invoking the LLM or summarizer, handle explicit eligibility checks.
    # Eligibility logic should only run when:
    # 1) The user explicitly asks about minimum/required scores (strictly
    #    determined by _is_eligibility_query()).
    # 2) A Tawjihi score is present in conversation memory.
    # 3) The retrieved context contains strong minimum markers (checked by
    #    _contains_minimum_marker()).
    # If these conditions hold and an explicit threshold is found, compare
    # the user's score to the threshold. If no threshold is found, skip
    # eligibility mode and continue with normal retrieval/LLM processing.
    if _is_eligibility_query(query):
        # Extract user's Tawjihi score from memory text.
        user_score_match = re.search(r"(\d{2,3})", memory_text or "")
        user_score_val: Optional[int] = None
        if user_score_match:
            try:
                user_score_val = int(user_score_match.group(1))
            except Exception:
                pass
        # Only proceed if user score exists and context has minimum marker
        if user_score_val is not None and _contains_minimum_marker(context_text):
            threshold = _extract_score_threshold(context_text)
            # Use eligibility response only when a threshold is actually found
            if threshold is not None:
                eligibility_resp = _format_eligibility_response(user_score_val, threshold, _is_arabic(query))
                out = {"answer": eligibility_resp, "sources": sources}
                QUERY_ANSWER_CACHE[query] = out
                return out

    # Try Groq with retrieved context. The system message instructs the model
    # to stay grounded in the provided university context and to keep answers
    # concise. We still post-process the output with our summarizer to limit
    # it to a reasonable number of bullet points.
    llm = groq_answer(query, context_text, memory_text=memory_text, chat_context=chat_context)
    if llm:
        # Shape the Groq response into shorter bullet points if needed
        shaped = _summarize_context(llm, lang_is_ar=_is_arabic(query), max_bullets=6)
        final_answer = shaped if shaped else llm
        out = {"answer": final_answer, "sources": sources}
        QUERY_ANSWER_CACHE[query] = out
        return out

    # Groq unavailable or failed: produce a clean summarized answer from the retrieved
    # context. This removes boilerplate like 'URL:' lines and limits output
    # length to a handful of sentences formatted as bullet points.
    summary = _summarize_context(context_text, lang_is_ar=_is_arabic(query), max_bullets=6)
    if summary:
        out = {"answer": summary, "sources": sources}
        QUERY_ANSWER_CACHE[query] = out
        return out

    # Fallback: trim the best chunk if summarization yields nothing
    resp = best["doc"]["text"].strip()
    if len(resp) > 900:
        resp = resp[:900].rsplit(" ", 1)[0] + "…"

    out = {"answer": resp, "sources": sources}
    QUERY_ANSWER_CACHE[query] = out
    return out


# ---------- SOURCES VIEWER (fallback) ----------
def safe_join_data_dir(rel_path: str) -> str:
    full = os.path.normpath(os.path.join(DATA_DIR, rel_path))
    if not full.startswith(os.path.normpath(DATA_DIR)):
        raise ValueError("Unsafe path")
    return full


def extract_url_from_text(text: str) -> Optional[str]:
    m = re.search(r"\bURL:\s*(https?://\S+)", text or "")
    return m.group(1) if m else None


def source_to_real_link(rel_path: str) -> Optional[str]:
    """
    Convert a saved chunk source like 'scraped/xxx.txt' into the original website URL
    by reading the first 'URL:' line in that file.
    """
    try:
        full_path = safe_join_data_dir(rel_path)
        if not os.path.exists(full_path):
            return None
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return extract_url_from_text(raw)
    except Exception:
        return None


@app.route("/source/<path:rel_path>", methods=["GET"])
def view_source(rel_path: str):
    """
    Fallback viewer:
    - If the saved file contains a URL: redirect to it
    - Else show the stored text
    """
    try:
        full_path = safe_join_data_dir(rel_path)
    except Exception:
        abort(404)

    if not os.path.exists(full_path):
        abort(404)

    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    url = extract_url_from_text(raw)
    if url:
        return redirect(url)

    return f"<pre style='white-space:pre-wrap;font-family:system-ui'>{raw}</pre>"


# ---------- SESSION HISTORY HELPERS ----------
MAX_HISTORY_LEN = 20


def _normalize_history() -> List[Dict[str, Any]]:
    hist = session.get("history", [])
    out: List[Dict[str, Any]] = []
    for item in hist:
        if isinstance(item, dict) and "q" in item and "a" in item:
            out.append({
                "q": item.get("q", ""),
                "a": item.get("a", ""),
                "sources": item.get("sources", []),
            })
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append({"q": item[0], "a": item[1], "sources": []})
    return out


def _build_chat_context(history: List[Dict[str, Any]], last_n: int = 4) -> str:
    recent = history[-last_n:]
    if not recent:
        return "(none)"
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['q']}")
        lines.append(f"Assistant: {turn['a']}")
    return "\n".join(lines)


def _summarize_context(text: str, lang_is_ar: bool, max_bullets: int = 8) -> str:
    """
    Create a short, bulleted summary from raw retrieved text when Groq is
    unavailable. This function cleans up unwanted lines (URLs, navigation
    instructions) and selects the first few substantive sentences to
    present as bullet points. Arabic questions will use the same bullet
    symbol, since the UI expects consistent list formatting in either
    language.

    Args:
        text: Combined text from top retrieved chunks.
        lang_is_ar: Whether the user question is in Arabic.
        max_bullets: Maximum number of bullet points to return.

    Returns:
        A bullet-point summary string or an empty string if no suitable
        sentences are found.
    """
    if not text:
        return ""
    # Drop lines starting with "URL:" or containing navigation hints
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines referencing URLs or navigation
        if line.lower().startswith("url:"):
            continue
        if "skip to content" in line.lower():
            continue
        cleaned_lines.append(line)
    cleaned_text = " ".join(cleaned_lines)
    # Split into sentences using punctuation, including Arabic question mark
    sentences = re.split(r"(?<=[.!؟\?])\s+", cleaned_text)
    bullets: List[str] = []
    for s in sentences:
        s = s.strip()
        # Skip if too short or too long or still contains navigation fragments
        if not s or len(s) < 15:
            continue
        if "skip to content" in s.lower():
            continue
        # Remove repeated headers (very short title-like phrases)
        bullets.append(s)
        if len(bullets) >= max_bullets:
            break
    if not bullets:
        return ""
    bullet_prefix = "• "  # Use same bullet symbol for both languages
    return "\n".join(bullet_prefix + b for b in bullets)


def _is_eligibility_query(q: str) -> bool:
    """
    Return True only if the user query explicitly asks about minimum or required
    Tawjihi scores or whether their own score is sufficient. This check is
    intentionally strict to avoid misclassifying generic admissions queries.

    Conditions:
    - English: contains phrases like "minimum", "cutoff", "required", "at least",
      "not less than", "no less than" together with words like "score",
      "tawjihi", "percentage", or "%".
    - English: phrases like "my score" or "score" combined with "enough",
      "sufficient", "eligible", or "qualify".
    - Arabic: contains "الحد الأدنى", "لا يقل عن", "معدل القبول", or "يشترط".
    - Arabic: contains "معدلي"/"علامتي"/"درجتي" combined with
      "كافي", "مناسب", "يكفي", or "بنفع".

    This function does NOT trigger for general application questions like
    "How can I apply?" or "What programs can I apply to?".
    """
    if not q:
        return False
    q_strip = q.strip()
    # Detect Arabic query by presence of Arabic unicode
    is_ar = bool(re.search(r"[\u0600-\u06FF]", q_strip))
    # English checks
    if not is_ar:
        q_low = q_strip.lower()
        # Rule 1: minimum-like patterns with score
        if re.search(r"(minimum|cutoff|required|at\s+least|not\s+less\s+than|no\s+less\s+than)", q_low) and \
           re.search(r"(score|tawjihi|percent|percentage|%)", q_low):
            return True
        # Rule 2: my score enough/eligible or a numeric score followed by "enough" etc.
        if re.search(r"(my\s+score|score)\s+.*(enough|sufficient|eligible|qualify)", q_low):
            return True
        if re.search(r"\b\d{2,3}\s*(?:%|percent)?\s*(enough|sufficient|eligible|qualify)", q_low):
            return True
        return False
    # Arabic checks
    # Rule 1: explicit minimum/cutoff words
    if any(kw in q_strip for kw in ["الحد الأدنى", "لا يقل عن", "معدل القبول", "يشترط"]):
        return True
    # Rule 2: asking if their score is enough/suitable
    if re.search(r"(معدلي|علامتي|درجتي).*(كافي|مناسب|يكفي|بنفع)", q_strip):
        return True
    # Also detect patterns like "85 كافي" or "85 يكفي" without using the words above
    if re.search(r"\b\d{2,3}\s*(?:٪|%|)\s*(كافي|يكفي|مناسب|بنفع)", q_strip):
        return True
    return False


def _extract_score_threshold(text: str) -> Optional[int]:
    """
    Extract a Tawjihi minimum threshold only when the number appears
    alongside clear indicators that it refers to a general secondary
    (Tawjihi) requirement. This function searches for numbers near
    keywords like "Tawjihi", "General Secondary", "الحد الأدنى",
    "لا يقل عن", "الثانوية العامة", etc. It ignores unrelated numbers
    (course grades, phone numbers, years, etc.). If no valid threshold
    is found, returns None.
    """
    if not text:
        return None
    candidates: List[int] = []
    # Define patterns that require the presence of Tawjihi or general secondary
    # words near minimum expressions. We capture the number as group 1.
    patterns = [
        # English: minimum/cutoff/required/at least/threshold followed by Tawjihi or general secondary mention
        r"(?i)(?:minimum|cutoff|required|at\s+least|not\s+less\s+than|no\s+less\s+than)\s+(?:tawjihi|general\s+secondary|score|percentage|percent)?\s*(?:of)?\s*(\d{2,3})(?:%|\b)",
        r"(?i)(?:tawjihi|general\s+secondary)\s*(?:minimum|required|cutoff|at\s+least|not\s+less\s+than|no\s+less\s+than)\s*(\d{2,3})(?:%|\b)",
        # Arabic: explicit minimum phrases with numbers
        r"(?:الحد\s+الأدنى\s*(?:لمعدل|لمجموع|لمعدل\s+التوجيهي)?\s*)(\d{2,3})(?:٪|%|\b)",
        r"(?:لا\s+يقل\s+عن)\s*(\d{2,3})(?:٪|%|\b)",
        r"(?:معدل\s+(?:التوجيهي|الثانوية\s+العامة)\s*(?:الحد\s+الأدنى|لا\s+يقل\s+عن|المطلوب|المطلوبة)\s*)(\d{2,3})(?:٪|%|\b)",
        r"(?:توجيهي|الثانوية\s+العامة)\s*(?:الحد\s+الأدنى|معدل\s+القبول|لا\s+يقل\s+عن|المطلوب|المطلوبة)\s*(\d{2,3})(?:٪|%|\b)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                num = int(m.group(1))
            except Exception:
                continue
            if 40 <= num <= 100:
                candidates.append(num)
    if candidates:
        return min(candidates)
    return None

# ---------- MINIMUM MARKER CHECK ----------
def _contains_minimum_marker(text: str) -> bool:
    """
    Return True if the context contains strong indicators of a minimum score
    requirement. This helps ensure eligibility logic only triggers when
    relevant information is actually present in the retrieved documents.

    English markers: "minimum", "cutoff", "at least", "not less than",
    "no less than", "required".
    Arabic markers: "الحد الأدنى", "لا يقل عن", "معدل القبول", "يشترط".
    """
    if not text:
        return False
    text_low = text.lower()
    english_terms = ["minimum", "cutoff", "at least", "not less than", "no less than", "required"]
    if any(term in text_low for term in english_terms):
        return True
    # Arabic check: we operate on raw text (not lowercased) to preserve Arabic
    if any(term in text for term in ["الحد الأدنى", "لا يقل عن", "معدل القبول", "يشترط"]):
        return True
    return False


def _format_eligibility_response(user_score: int, threshold: Optional[int], lang_is_ar: bool) -> str:
    """
    Compose a response regarding eligibility based on the user's score and the
    extracted threshold. If the threshold is None, acknowledge that it is
    unspecified. Otherwise, inform the user whether their score meets the
    requirement without making further inferences.
    """
    if threshold is None:
        return ("الحد الأدنى لمعدل التوجيهي غير مذكور صراحة على موقع الجامعة." if lang_is_ar
                else "The exact minimum score is not specified on the university website.")
    # Compare scores
    if user_score >= threshold:
        return (f"معدلك {user_score}% يساوي أو يتجاوز الحد الأدنى المطلوب وهو {threshold}% لذا يمكنك التقديم." if lang_is_ar
                else f"Your Tawjihi score of {user_score}% meets or exceeds the minimum requirement of {threshold}%.")
    else:
        return (f"معدلك {user_score}% أقل من الحد الأدنى المطلوب وهو {threshold}%، لذلك قد لا تكون مؤهلاً." if lang_is_ar
                else f"Your Tawjihi score of {user_score}% is below the minimum requirement of {threshold}%, so you may not be eligible.")


def _is_question(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    low = s.lower()
    if "?" in s or "؟" in s:
        return True
    return any(low.startswith(w) for w in [
        "what", "when", "where", "which", "who", "whom",
        "why", "how", "can", "do", "does", "is", "are",
        "هل", "ما", "ماذا", "متى", "أين", "لماذا", "كيف",
    ])


def _make_direct_source_links(sources: List[str]) -> List[Dict[str, str]]:
    """
    Convert file sources to direct website URLs (preferred),
    fallback to /source/<file> if URL not found.
    """
    links: List[Dict[str, str]] = []
    seen = set()

    for s in sources:
        url = source_to_real_link(s)
        href = url if url else f"/source/{s}"  # fallback

        if href in seen:
            continue
        seen.add(href)

        # Extract domain name for display (e.g., alquds.edu) when URL present
        if url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                # strip leading www.
                if domain.startswith("www."):
                    domain = domain[4:]
                label = domain
            except Exception:
                label = url
        else:
            label = s
        links.append({"label": label, "href": href})

    return links


# ---------- FLASK ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def chat():
    history = _normalize_history()
    session["history"] = history

    last_question = ""
    answer_text = ""
    answer_sources: List[Dict[str, str]] = []

    if request.method == "POST":
        last_question = (request.form.get("question") or "").strip()

        extracted = extract_memory_facts(last_question)
        mem_old = session.get("memory", {}) or {}
        mem = mem_old.copy()
        mem.update(extracted)
        session["memory"] = mem

        new_score = "tawjihi_score" in extracted and extracted.get("tawjihi_score") != mem_old.get("tawjihi_score")
        new_name = "name" in extracted and extracted.get("name") != mem_old.get("name")

        # If user only shared info and didn't ask a question -> acknowledge (no retrieval)
        if (new_score or new_name) and (not _is_question(last_question)):
            lang_ar = _is_arabic(last_question)
            responses = []
            if new_name:
                nm = extracted.get("name")
                responses.append(f"تشرفنا بك، {nm}." if lang_ar else f"Nice to meet you, {nm}.")
            if new_score:
                sc = extracted.get("tawjihi_score")
                responses.append(f"مبروك، تم تسجيل معدل التوجيهي {sc}%." if lang_ar else f"Congratulations — I saved your Tawjihi score ({sc}%).")
            answer_text = " ".join(responses).strip() or ("تم." if lang_ar else "Got it.")
            answer_sources = []
        else:
            chat_context = _build_chat_context(history, last_n=4)
            mem_text = memory_to_text(mem, lang_is_ar=_is_arabic(last_question))

            aug_q = last_question
            score = mem.get("tawjihi_score")
            if score:
                low = last_question.lower()
                if ("my score" in low) or ("with my score" in low) or ("score" in low) or ("معدلي" in last_question) or ("علامتي" in last_question):
                    aug_q = f"{last_question} (Tawjihi score: {score})"

            result = answer_query(aug_q, memory_text=mem_text, chat_context=chat_context)
            answer_text = result["answer"]
            sources = result.get("sources", [])
            answer_sources = _make_direct_source_links(sources)

        if last_question:
            history.append({"q": last_question, "a": answer_text, "sources": answer_sources})
            session["history"] = history[-MAX_HISTORY_LEN:]
            session.modified = True

    return render_template(
        "chat.html",
        history=session.get("history", []),
        last_question=last_question,
        answer=answer_text,
        answer_sources=answer_sources,
    )


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()

    history = _normalize_history()
    session["history"] = history

    if not q:
        ans = "Please type a question."
        return jsonify({"answer": ans, "reply": ans, "message": ans, "sources": [], "ok": False}), 200

    extracted = extract_memory_facts(q)
    mem_old = session.get("memory", {}) or {}
    mem = mem_old.copy()
    mem.update(extracted)
    session["memory"] = mem

    new_score = "tawjihi_score" in extracted and extracted.get("tawjihi_score") != mem_old.get("tawjihi_score")
    new_name = "name" in extracted and extracted.get("name") != mem_old.get("name")

    try:
        if (new_score or new_name) and (not _is_question(q)):
            lang_ar = _is_arabic(q)
            responses = []
            if new_name:
                nm = extracted.get("name")
                responses.append(f"تشرفنا بك، {nm}." if lang_ar else f"Nice to meet you, {nm}.")
            if new_score:
                sc = extracted.get("tawjihi_score")
                responses.append(f"مبروك، تم تسجيل معدل التوجيهي {sc}%." if lang_ar else f"Congratulations — I saved your Tawjihi score ({sc}%).")
            ans = " ".join(responses).strip() or ("تم." if lang_ar else "Got it.")
            source_links: List[Dict[str, str]] = []
        else:
            chat_context = _build_chat_context(history, last_n=4)
            mem_text = memory_to_text(mem, lang_is_ar=_is_arabic(q))

            aug_q = q
            score = mem.get("tawjihi_score")
            if score:
                low = q.lower()
                if ("my score" in low) or ("with my score" in low) or ("score" in low) or ("معدلي" in q) or ("علامتي" in q):
                    aug_q = f"{q} (Tawjihi score: {score})"

            result = answer_query(aug_q, memory_text=mem_text, chat_context=chat_context)
            ans = result["answer"]
            sources = result.get("sources", [])
            source_links = _make_direct_source_links(sources)

    except Exception as e:
        ans = f"⚠️ Error: {type(e).__name__}: {e}"
        source_links = []

    history.append({"q": q, "a": ans, "sources": source_links})
    session["history"] = history[-MAX_HISTORY_LEN:]
    session.modified = True

    return jsonify({
        "answer": ans,
        "reply": ans,
        "message": ans,
        "sources": source_links,
        "ok": True
    }), 200


@app.route("/reset", methods=["POST"])
def reset_chat():
    session.pop("history", None)
    session.pop("memory", None)
    return ("", 204)


# ---------- MAIN ----------
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True, use_reloader=False)
