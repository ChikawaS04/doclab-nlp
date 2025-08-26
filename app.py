import os, io, re, unicodedata
from typing import Tuple, List, Dict, Set

from flask import Flask, jsonify, request
from flask_cors import CORS
import pdfplumber
import re

# --- spaCy (safe load with fallback) ---
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    def nlp(text: str):
        return _NLP(text)
    SPACY_OK = True
except Exception:
    # Minimal fallback: split sentences on punctuation/newlines; no NER
    SPACY_OK = False
    _sent_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|[\r\n]+")
    def nlp(text: str):
        class _Doc:
            def __init__(self, t: str):
                self.text = t
                # very rough "sentences"
                self.sents = [type("S", (), {"text": s}) for s in _sent_splitter.split(t) if s.strip()]
                self.ents = []  # no NER in fallback
            def __iter__(self):
                return iter(())
        return _Doc(text)

app = Flask(__name__)
CORS(app)

# -----------------------------
# Normalization & Text Helpers
# -----------------------------
def fix_spacing(s: str) -> str:
    """
    Repairs common 'glued' text artifacts after PDF extraction while trying
    not to mangle emails/URLs. Final pass collapses multi-spaces.
    """
    if not s:
        return s

    # Join hyphenated line-breaks: 'inter-\nnational' -> 'international'
    s = re.sub(r'-\s*\n\s*', '', s)

    # Turn hard line breaks into spaces (preserve paragraphs loosely)
    s = re.sub(r'\s*\n\s*', ' ', s)

    # Insert missing spaces after punctuation when the next char is a letter
    # e.g., "amount,which" -> "amount, which"
    s = re.sub(r'(?<=[,;:])(?=[A-Za-z])', ' ', s)

    # Insert missing spaces between numbers and letters in both directions
    # e.g., "123ABC" -> "123 ABC", "rate5percent" -> "rate 5 percent"
    s = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', s)
    s = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', s)

    # Insert missing space after sentence period when next char is uppercase letter
    # e.g., "Agreement.The" -> "Agreement. The"
    s = re.sub(r'(?<=[A-Za-z0-9])\.(?=[A-Z])', '. ', s)

    # Normalize weird encodings already seen (dashes, NBSP)
    s = (s
         .replace("ΓÇô", "–")
         .replace("â€“", "–")
         .replace("â€”", "—")
         .replace("\u00A0", " "))

    # Collapse any runs of whitespace to a single space
    s = re.sub(r'[ \t]{2,}', ' ', s).strip()
    return s

def normalize_text(s: str) -> str:
    if not s:
        return s
    return fix_spacing(s)

# -----------------------------
# File Readers
# -----------------------------
def read_pdf(bytes_data: bytes) -> tuple[str, int]:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
        for page in pdf.pages:
            # Preserve more natural spacing/flow
            t = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True) or ""
            if t.strip():
                text_chunks.append(t)
        return ("\n".join(text_chunks).strip(), len(pdf.pages))

def read_docx(bytes_data: bytes) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(io.BytesIO(bytes_data))
    return "\n".join(p.text for p in doc.paragraphs).strip()

# -----------------------------
# Heuristics: doc type
# -----------------------------
def detect_doc_type(text: str, filename: str) -> str:
    name = (filename or "").lower()
    t = (text or "").lower()

    # very light signals
    if any(k in name for k in ["resume", "cv"]) or any(k in t for k in ["education", "experience", "skills"]):
        return "Resume"
    if any(k in name for k in ["invoice", "receipt"]) or any(k in t for k in ["invoice #", "total due", "amount due", "subtotal"]):
        return "Invoice"
    if "non-disclosure" in name or "nda" in name or any(k in t for k in ["non-disclosure", "confidential information", "receiving party", "disclosing party"]):
        return "NDA"
    if any(k in name for k in ["contract", "agreement"]) or any(k in t for k in ["agreement", "governing law", "term and termination"]):
        return "Contract"
    if any(k in name for k in ["offer", "employment"]) or any(k in t for k in ["employee", "employer", "salary", "benefits"]):
        return "HR"
    if any(k in name for k in ["po", "purchase-order"]) or any(k in t for k in ["purchase order", "vendor", "ship to"]):
        return "Procurement"
    if "report" in name:
        return "Report"
    if "reference" in name or "references" in t:
        return "Reference"
    return "Unknown"

# -----------------------------
# Summarization
# -----------------------------
SECTION_KEYWORDS: Dict[str, Set[str]] = {
    # legal / generic
    "parties": {"party", "parties", "disclosing", "receiving", "company", "entity", "client", "vendor"},
    "purpose": {"purpose", "objective", "collaboration", "evaluation", "scope"},
    "confidential": {"confidential", "information", "data", "materials", "trade", "secret"},
    "obligations": {"agree", "duty", "obligation", "maintain", "protect", "disclose", "notify", "return", "destroy"},
    "term": {"term", "duration", "effective", "effective date", "expiration", "terminate", "termination", "renewal"},
    "governing": {"governing", "law", "jurisdiction", "state", "country", "venue"},
    "signatures": {"signature", "execute", "sign", "date", "witness", "counterpart", "agreement"},
    # finance / HR / procurement extras
    "payment": {"payment", "price", "fee", "charge", "invoice", "amount", "compensation"},
    "compliance": {"compliance", "warranty", "indemnify", "indemnification", "liability", "limitation"},
}

def summarize(text: str, max_sentences: int = 5, max_chars: int = 1400) -> str:
    """
    Frequency-based extractive summary with section coverage and mild position bias.
    """
    if not text:
        return ""

    doc = nlp(text)
    # collect non-empty sentences
    sents = [s for s in getattr(doc, "sents", []) if getattr(s, "text", "").strip()]
    if not sents:
        return text[:max_chars]

    # build frequencies if spaCy tokens exist; else fallback to first N sents
    freqs: Dict[str, float] = {}
    tokens = list(getattr(doc, "__iter__", lambda: iter(()))())
    if tokens:
        for t in tokens:
            if getattr(t, "is_stop", False) or getattr(t, "is_punct", False) or getattr(t, "like_num", False) or getattr(t, "is_space", False):
                continue
            lemma = getattr(t, "lemma_", None) or str(getattr(t, "text", "")).lower()
            lemma = (lemma or "").lower().strip()
            if lemma:
                freqs[lemma] = freqs.get(lemma, 0.0) + 1.0

    if not freqs:
        out, total = [], 0
        for s in sents:
            nxt = s.text.strip()
            if total + (1 if out else 0) + len(nxt) > max_chars: break
            out.append(nxt if not out else " " + nxt)
            total += (1 if out else 0) + len(nxt)
            if len(out) >= max_sentences: break
        return "".join(out)

    # normalize freqs
    max_f = max(freqs.values())
    for k in list(freqs.keys()):
        freqs[k] /= max_f

    # score sentences
    scored: List[Tuple[int, float, str, Set[str]]] = []
    for idx, s in enumerate(sents):
        # collect lemmas
        words = []
        for t in getattr(s, "__iter__", lambda: iter(()))():
            if getattr(t, "is_stop", False) or getattr(t, "is_punct", False) or getattr(t, "like_num", False) or getattr(t, "is_space", False):
                continue
            lemma = getattr(t, "lemma_", None) or str(getattr(t, "text", "")).lower()
            lemma = (lemma or "").lower().strip()
            if lemma:
                words.append(lemma)

        if not words:
            continue

        base_score = sum(freqs.get(w, 0.0) for w in words)
        length_norm = max(1.0, len(words) ** 0.85)

        # section hits
        secs_hit: Set[str] = set()
        wset = set(words)
        for sec, kws in SECTION_KEYWORDS.items():
            if wset.intersection(kws):
                secs_hit.add(sec)

        # position bias: earlier sentences get a tiny boost
        pos_bias = 1.0 + max(0.0, (len(sents) - idx) / len(sents) * 0.05)

        # section boost if any hit
        section_boost = 1.3 if secs_hit else 1.0
        score = (base_score / length_norm) * section_boost * pos_bias
        scored.append((idx, score, s.text.strip(), secs_hit))

    if not scored:
        txt = (sents[0].text + (" " + sents[1].text if len(sents) > 1 else "")).strip()
        return txt[:max_chars]

    scored.sort(key=lambda x: x[1], reverse=True)

    # ensure section coverage
    chosen: List[Tuple[int, float, str, Set[str]]] = []
    seen_idx: Set[int] = set()
    for sec in SECTION_KEYWORDS.keys():
        for tup in scored:
            idx, _, _, secs = tup
            if idx in seen_idx: continue
            if sec in secs:
                chosen.append(tup); seen_idx.add(idx)
                break
        if len(chosen) >= max_sentences: break

    # fill remaining
    for tup in scored:
        if len(chosen) >= max_sentences: break
        idx = tup[0]
        if idx not in seen_idx:
            chosen.append(tup); seen_idx.add(idx)

    # restore original order and pack under char budget
    chosen.sort(key=lambda x: x[0])
    out, total = [], 0
    for _, _, txt, _ in chosen:
        add = ((" " if out else "") + txt)
        if total + len(add) > max_chars: break
        out.append(add); total += len(add)

    if not out:
        out = [scored[0][2][:max_chars]]

    return "".join(out)

# -----------------------------
# Entities → fields
# -----------------------------
def extract_fields(text: str) -> List[Dict[str, str]]:
    doc = nlp(text or "")
    out: List[Dict[str, str]] = []
    seen = set()
    for ent in getattr(doc, "ents", []):
        if getattr(ent, "label_", "") in {"PERSON", "ORG", "GPE", "DATE", "MONEY"}:
            label = ent.label_.strip()
            value = str(ent.text).strip()
            if not label or not value:
                continue
            key = (label, value)
            if key in seen:
                continue
            seen.add(key)
            out.append({"label": label, "text": value})
            if len(out) >= 50:
                break
    return out

# -----------------------------
# Routes
# -----------------------------
@app.post("/process")
def process():
    if "file" not in request.files:
        return jsonify(error="No file part 'file'"), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    raw = f.read()

    try:
        if ext == ".pdf":
            text, pages = read_pdf(raw)
        elif ext == ".docx":
            text, pages = read_docx(raw), 1
        elif ext == ".txt":
            text, pages = raw.decode("utf-8", errors="ignore"), 1
        else:
            return jsonify(error=f"Unsupported file type '{ext}' (PDF/DOCX/TXT only for now)"), 415
    except Exception as e:
        return jsonify(error=f"Failed to parse file: {str(e)}"), 500

    text = normalize_text(text)
    doc_type = detect_doc_type(text, filename)

    # title: first non-empty line with letters; fallback to filename
    first_line = next((ln.strip() for ln in (text or "").splitlines() if any(ch.isalpha() for ch in ln)), filename)
    title = first_line or filename

    summary = summarize(text)
    entities = extract_fields(text)

    return jsonify({
        "title": title,
        "summary": summary or "",
        "entities": entities,          # [{"label": "...", "text": "..."}]
        "docType": doc_type,           # top-level convenience
        "pages": pages,
        "meta": {
            "docType": doc_type,       # Java reads this into NlpMeta.docType
            "filename": filename,
            "size": len(raw),
            "nlp": "spaCy" if SPACY_OK else "fallback"
        },
    }), 200

@app.get("/health")
def health():
    return jsonify(status="ok", service="doclab-nlp", version="0.4.0", spacy=SPACY_OK)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # threaded True helps with quick local testing
    app.run(host="0.0.0.0", port=port, threaded=True)
