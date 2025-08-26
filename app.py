import io, os, re
from typing import Tuple, List, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS
import pdfplumber
import spacy
from docx import Document as DocxDocument

# ---- spaCy (load once) -------------------------------------------------------
# If the model isn't present, we'll fall back to a tiny rule-based splitter.
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    nlp = None
    SPACY_OK = False

app = Flask(__name__)
CORS(app)


# ---- text helpers -------------------------------------------------------------
def _fix_spacing(s: str) -> str:
    """Repair common PDF spacing glitches and hyphenated line breaks."""
    if not s:
        return s

    # join words split by line hyphenation: inter-\nnational -> international
    s = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', s)

    # ensure a space after punctuation when followed by a letter/number
    s = re.sub(r'([a-zA-Z0-9][,;:])(?!\s)', r'\1 ', s)
    # ensure a space after period when next token starts a new sentence
    s = re.sub(r'([a-zA-Z0-9])\.(?=[A-Z])', r'\1. ', s)

    # digit/letter boundaries
    s = re.sub(r'(\d)([A-Za-z])', r'\1 \2', s)
    s = re.sub(r'([A-Za-z])(\d)', r'\1 \2', s)

    # collapse multi-spaces and normalize NBSP
    s = s.replace("\u00A0", " ")
    s = re.sub(r'[ \t]+', ' ', s)

    # collapse multiple newlines to at most 2
    s = re.sub(r'\n{3,}', '\n\n', s)

    return s.strip()


def _normalize_text(s: str) -> str:
    if not s:
        return s
    s = (s
         .replace("ΓÇô", "–")
         .replace("â€“", "–")
         .replace("â€”", "—")
         .replace("â€˜", "‘")
         .replace("â€™", "’")
         .replace("â€œ", "“")
         .replace("â€�", "”"))
    s = _fix_spacing(s)
    return s


# ---- file readers -------------------------------------------------------------
def read_pdf(data: bytes) -> Tuple[str, int]:
    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for p in pdf.pages:
            # a little tolerance helps preserve spacing
            t = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
            if t.strip():
                text_parts.append(t)
        return "\n".join(text_parts).strip(), len(pdf.pages)


def read_docx(data: bytes) -> str:
    d = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs).strip()


# ---- simple type detection ----------------------------------------------------
def detect_doc_type(text: str, filename: str) -> str:
    name = (filename or "").lower()
    t = (text or "").lower()
    if "loan" in name or "promissory" in name or any(k in t for k in ["borrower", "lender", "principal", "installment"]):
        return "Contract"
    if any(k in name for k in ["invoice", "receipt"]) or any(k in t for k in ["invoice #", "total due", "amount due"]):
        return "Invoice"
    if any(k in name for k in ["resume", "cv"]) or any(k in t for k in ["education", "experience", "skills"]):
        return "Resume"
    if "reference" in name or "references" in t:
        return "Reference"
    if "report" in name:
        return "Report"
    return "Unknown"


# ---- summarization ------------------------------------------------------------
SECTION_KEYWORDS = {
    "parties": {"party", "parties", "disclosing", "receiving", "lender", "borrower", "company", "entity"},
    "purpose": {"purpose", "objective", "loan", "collaboration", "evaluation"},
    "confidential": {"confidential", "information", "data", "materials", "trade", "secret"},
    "terms": {"term", "duration", "installment", "principal", "interest", "repayment", "schedule", "effective", "expiration", "terminate", "termination"},
    "governing": {"governing", "law", "jurisdiction", "state", "country"},
    "signatures": {"signature", "execute", "date", "witness", "agreement"}
}

def _sentences(text: str) -> List[str]:
    if SPACY_OK and nlp:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # tiny fallback: split on period/end punctuation
    raw = re.split(r'(?<=[\.!?])\s+', text)
    return [r.strip() for r in raw if r.strip()]

def summarize(text: str, max_sentences: int = 5, max_chars: int = 1400) -> str:
    if not text:
        return ""
    sents = _sentences(text)
    if not sents:
        return text[:max_chars]

    # frequency scoring (lemma-based when spaCy available)
    freqs: Dict[str, float] = {}
    if SPACY_OK and nlp:
        doc = nlp(text)
        for t in doc:
            if t.is_stop or t.is_punct or t.like_num or t.is_space:
                continue
            lemma = t.lemma_.lower().strip()
            if lemma:
                freqs[lemma] = freqs.get(lemma, 0.0) + 1.0
    else:
        # crude token freq fallback
        for tok in re.findall(r"[A-Za-z]+", text.lower()):
            freqs[tok] = freqs.get(tok, 0.0) + 1.0

    if not freqs:
        return " ".join(sents[:max_sentences])[:max_chars]

    max_f = max(freqs.values())
    for k in list(freqs.keys()):
        freqs[k] /= max_f

    scored: List[tuple] = []
    for idx, s in enumerate(sents):
        toks = re.findall(r"[A-Za-z]+", s.lower())
        if not toks:
            continue
        base = sum(freqs.get(t, 0.0) for t in toks)
        length_norm = max(1.0, len(toks) ** 0.85)

        # section boost
        sset = set(toks)
        sections_hit = {sec for sec, kws in SECTION_KEYWORDS.items() if sset.intersection(kws)}
        boost = 1.3 if sections_hit else 1.0
        score = (base / length_norm) * boost
        scored.append((idx, score, s, sections_hit))

    if not scored:
        return " ".join(sents[:max_sentences])[:max_chars]

    scored.sort(key=lambda x: x[1], reverse=True)

    # ensure coverage where possible
    chosen, seen, covered = [], set(), set()
    for sec in SECTION_KEYWORDS.keys():
        for tup in scored:
            i, _, txt, hit = tup
            if i in seen:
                continue
            if sec in hit:
                chosen.append(tup)
                seen.add(i)
                covered.add(sec)
                break
        if len(chosen) >= max_sentences:
            break

    for tup in scored:
        if len(chosen) >= max_sentences:
            break
        i = tup[0]
        if i not in seen:
            chosen.append(tup)
            seen.add(i)

    chosen.sort(key=lambda x: x[0])

    out, total = [], 0
    for _, _, txt, _ in chosen:
        add = (" " if out else "") + txt
        if total + len(add) > max_chars:
            break
        out.append(add)
        total += len(add)

    if not out:
        out = [scored[0][2][:max_chars]]

    return "".join(out)


# ---- entity extraction ---------------------------------------------------------
def extract_fields(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        if SPACY_OK and nlp:
            doc = nlp(text)
            out = []
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY"}:
                    # keys exactly as Java expects
                    out.append({"label": ent.label_, "text": ent.text})
            return out[:50]
        else:
            # tiny fallback – just dates and money-like patterns
            out = []
            for m in re.finditer(r"\b(?:\w+\s\d{1,2},\s\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b", text):
                out.append({"label": "DATE", "text": m.group(0)})
            for m in re.finditer(r"\$\s?\d[\d,]*(?:\.\d{2})?", text):
                out.append({"label": "MONEY", "text": m.group(0)})
            return out[:50]
    except Exception:
        return []


# ---- API ----------------------------------------------------------------------
@app.post("/process")
def process():
    try:
        if 'file' not in request.files:
            return jsonify(error="No file part 'file'"), 400

        f = request.files['file']
        if not f or f.filename == "":
            return jsonify(error="Empty filename"), 400

        filename = f.filename
        ext = os.path.splitext(filename)[1].lower()
        raw = f.read() or b""

        if ext == ".pdf":
            text, pages = read_pdf(raw)
        elif ext == ".docx":
            text, pages = read_docx(raw), 1
        elif ext == ".txt":
            text, pages = raw.decode("utf-8", errors="ignore"), 1
        else:
            return jsonify(error=f"Unsupported file type '{ext}' (PDF/DOCX/TXT only)"), 415

        text = _normalize_text(text)
        doc_type = detect_doc_type(text, filename)

        # title = first non-empty line with letters; otherwise filename
        first_line = next((ln.strip() for ln in (text or "").splitlines() if any(ch.isalpha() for ch in ln)), filename)
        title = first_line or filename

        summary = summarize(text)
        entities = extract_fields(text)

        # IMPORTANT: never return nulls
        return jsonify({
            "title": title or filename,
            "summary": summary or "",
            "entities": entities or [],
            "docType": doc_type or "Unknown",
            "pages": int(pages or 1),
            "meta": {
                "docType": doc_type or "Unknown",
                "filename": filename,
                "size": int(len(raw))
            }
        }), 200

    except Exception as e:
        # Return a safe JSON; your Java side will capture lastError if needed
        return jsonify(error=f"Unhandled server error: {type(e).__name__}: {str(e)}"), 500


@app.get("/health")
def health():
    return jsonify(status="ok", service="doclab-nlp", version="0.4.0")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
