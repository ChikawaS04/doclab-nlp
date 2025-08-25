import os, io
from flask import Flask, jsonify, request
from flask_cors import CORS
import pdfplumber
import spacy
from docx import Document as DocxDocument

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)

# --- helpers ---

def read_pdf(bytes_data: bytes) -> tuple[str, int]:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
        for page in pdf.pages:
            # Slightly more tolerant extraction to reduce truncation
            t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            if t.strip():
                text_chunks.append(t)
        return ("\n".join(text_chunks).strip(), len(pdf.pages))

def read_docx(bytes_data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(bytes_data))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def detect_doc_type(text: str, filename: str) -> str:
    name = filename.lower()
    t = (text or "").lower()
    if any(k in name for k in ["invoice", "receipt"]) or any(k in t for k in ["invoice #", "total due", "amount due"]):
        return "Invoice"
    if any(k in name for k in ["resume", "cv"]) or any(k in t for k in ["education", "experience", "skills"]):
        return "Resume"
    if "reference" in name or "references" in t:
        return "Reference"
    if "report" in name:
        return "Report"
    return "Unknown"

def normalize_text(s: str) -> str:
    if not s:
        return s
    # quick fix for weird dash artifacts etc.
    return (s
            .replace("ΓÇô", "–")
            .replace("â€“", "–")
            .replace("â€”", "—")
            .replace("\u00A0", " ")  # non-breaking space
            ).strip()

def summarize(text: str, max_sentences: int = 4, max_chars: int = 1200) -> str:
    """
    Frequency-based extractive summary for legal/financial documents.
    Returns up to `max_sentences` sentences, capped at `max_chars`, prioritizing key sections.
    """
    if not text:
        return ""

    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except AttributeError:
        return text[:max_chars]  # Fallback if spaCy not available

    doc = nlp(text)

    # Collect sentences (strip empties)
    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return text[:max_chars]

    # Define keywords for legal/financial sections (generic)
    section_keywords = {
        "parties": ["party", "parties", "disclosing", "receiving", "company", "entity"],
        "purpose": ["purpose", "objective", "collaboration", "evaluation"],
        "confidential": ["confidential", "information", "data", "materials", "trade secret"],
        "obligations": ["agree", "duty", "obligation", "maintain", "protect"],
        "term": ["term", "duration", "effective", "expiration", "termination"],
        "governing": ["governing", "law", "jurisdiction", "state", "country"],
        "signatures": ["signature", "execute", "date", "witness", "agreement"]
    }

    # Build a frequency table for content words, boosting section-specific terms
    freqs = {}
    for token in doc:
        if (
                token.is_stop
                or token.is_punct
                or token.like_num
                or token.is_space
                or not token.text.strip()
        ):
            continue
        key = token.lemma_.lower()
        freqs[key] = freqs.get(key, 0) + 1
        # Boost frequency for section-related keywords
        for section, keywords in section_keywords.items():
            if key in keywords:
                freqs[key] *= 2  # Double weight for relevance

    if not freqs:
        # Fallback: first few sentences
        out = []
        for s in sents:
            if len(" ".join(out)) + len(s.text) > max_chars:
                break
            out.append(s.text.strip())
            if len(out) >= max_sentences:
                break
        return " ".join(out)

    # Normalize frequencies
    max_f = max(freqs.values())
    for k in freqs:
        freqs[k] = freqs[k] / max_f

    # Score each sentence by normalized sum of token freqs (with section priority)
    scored = []
    for idx, s in enumerate(sents):
        tokens = [t for t in s if t.lemma_.lower() in freqs]
        if not tokens:
            continue
        score = sum(freqs[t.lemma_.lower()] for t in tokens)
        # Apply length normalization and section boost
        denom = max(1.0, (len(tokens)) ** 0.85)
        # Boost score if sentence contains section keywords
        section_boost = 1.0
        for section, keywords in section_keywords.items():
            if any(k in [t.lemma_.lower() for t in s] for k in keywords):
                section_boost = 1.5
                break
        scored.append((idx, score * section_boost / denom, s.text.strip()))

    if not scored:
        return (sents[0].text + " " + (sents[1].text if len(sents) > 1 else "")).strip()[:max_chars]

    # Pick top-N by score, then restore original order
    scored.sort(key=lambda x: x[1], reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda x: x[0])

    # Enforce char cap without cutting sentences mid-word, ensuring section coverage
    out = []
    total = 0
    for _, _, s_text in chosen:
        # Check if adding this sentence exceeds limit
        if total + len(s_text) + (1 if out else 0) > max_chars:
            # Try to include at least one sentence per major section if possible
            if any(any(k in s_text.lower() for k in keywords) for keywords in section_keywords.values()):
                if total + len(s_text) <= 2000:  # Allow up to 2000 as a soft cap
                    out.append(s_text)
                    total += len(s_text) + (1 if out else 0)
            break
        out.append(s_text)
        total += len(s_text) + (1 if out else 0)

    # If we somehow added nothing (extreme short cap), take the best single sentence
    if not out:
        out = [scored[0][2][:max_chars]]

    return " ".join(out)


def extract_fields(text: str):
    fields = []
    if not text:
        return fields
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY"}:
            # IMPORTANT: use keys expected by Java DTOs (label/text)
            fields.append({"label": ent.label_, "text": ent.text})
    # keep it light
    return fields[:50]

# --- routes ---

@app.post("/process")
def process():
    if 'file' not in request.files:
        return jsonify(error="No file part 'file'"), 400

    f = request.files['file']
    if f.filename == "":
        return jsonify(error="Empty filename"), 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    raw = f.read()  # bytes

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

    # Better title: prefer first non-empty line containing letters; fallback to filename
    first_line = next((ln.strip() for ln in (text or "").splitlines() if any(ch.isalpha() for ch in ln)), filename)
    title = first_line or filename

    summary = summarize(text)
    entities = extract_fields(text)

    # NOTE: include title; keep docType in meta (what your Java expects) and also top-level for convenience
    return jsonify({
        "title": title,
        "summary": summary or "",
        "entities": entities,               # [{"label": "...", "text": "..."}]
        "docType": doc_type,                # top-level (optional)
        "pages": pages,
        "meta": {
            "docType": doc_type,            # <-- Java NlpMeta.docType will see this
            "filename": filename,
            "size": len(raw)
        }
    }), 200

@app.get("/health")
def health():
    return jsonify(status="ok", service="doclab-nlp", version="0.3.0")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
