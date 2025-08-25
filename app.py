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

SECTION_KEYWORDS = {
    "parties": {"party", "parties", "disclosing", "receiving", "company", "entity"},
    "purpose": {"purpose", "objective", "collaboration", "evaluation"},
    "confidential": {"confidential", "information", "data", "materials", "trade", "secret"},
    "obligations": {"agree", "duty", "obligation", "maintain", "protect", "disclose"},
    "term": {"term", "duration", "effective", "expiration", "terminate", "termination"},
    "governing": {"governing", "law", "jurisdiction", "state", "country"},
    "signatures": {"signature", "execute", "date", "witness", "agreement"}
}

def summarize(text: str, max_sentences: int = 5, max_chars: int = 1400) -> str:
    """
    Frequency-based extractive summary tuned for legal/financial docs.
    Ensures (best-effort) section coverage, normalized by sentence length,
    and capped to `max_chars`.
    """
    if not text:
        return ""

    doc = nlp(text)

    # Collect non-empty sentences
    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return text[:max_chars]

    # Build frequency table on lemmas (skip stop/punct/nums)
    freqs: dict[str, float] = {}
    for t in doc:
        if t.is_stop or t.is_punct or t.like_num or t.is_space:
            continue
        lemma = t.lemma_.lower().strip()
        if not lemma:
            continue
        freqs[lemma] = freqs.get(lemma, 0.0) + 1.0

    if not freqs:
        out = []
        for s in sents:
            nxt = ("" if not out else " ") + s.text.strip()
            if len("".join(out)) + len(nxt) > max_chars or len(out) >= max_sentences:
                break
            out.append(nxt)
        return "".join(out)

    max_f = max(freqs.values())
    for k in list(freqs):
        freqs[k] /= max_f  # normalize to [0,1]

    # Precompute sentence keyword hits + scores
    scored = []  # (idx, score, text, sections_hit:set)
    for idx, s in enumerate(sents):
        lemmas = [t.lemma_.lower() for t in s if not (t.is_stop or t.is_punct or t.like_num or t.is_space)]
        if not lemmas:
            continue

        base_score = sum(freqs.get(l, 0.0) for l in lemmas)
        length_norm = max(1.0, len(lemmas) ** 0.85)

        sections_hit = set()
        sent_lemmas = set(lemmas)
        for sec, kws in SECTION_KEYWORDS.items():
            if sent_lemmas.intersection(kws):
                sections_hit.add(sec)

        # Boost if we hit any section keywords
        boost = 1.3 if sections_hit else 1.0
        score = (base_score / length_norm) * boost

        scored.append((idx, score, s.text.strip(), sections_hit))

    if not scored:
        # Safe fallback: first couple of sentences
        txt = (sents[0].text + (" " + sents[1].text if len(sents) > 1 else "")).strip()
        return txt[:max_chars]

    # Sort high → low by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # 1) Try to cover sections: pick best sentence per section if available
    chosen: list[tuple[int,float,str,set]] = []
    seen_idx = set()
    covered_sections = set()
    for sec in SECTION_KEYWORDS.keys():
        for tup in scored:
            idx, _, txt, secs = tup
            if idx in seen_idx:
                continue
            if sec in secs:
                chosen.append(tup)
                seen_idx.add(idx)
                covered_sections.add(sec)
                break
        if len(chosen) >= max_sentences:
            break

    # 2) Fill remaining slots by score
    for tup in scored:
        if len(chosen) >= max_sentences:
            break
        idx = tup[0]
        if idx not in seen_idx:
            chosen.append(tup)
            seen_idx.add(idx)

    # Restore original text order
    chosen.sort(key=lambda x: x[0])

    # 3) Pack into char budget
    out_parts = []
    total = 0
    for _, _, txt, _ in chosen:
        sep = "" if not out_parts else " "
        if total + len(sep) + len(txt) > max_chars:
            break
        out_parts.append(sep + txt)
        total += len(sep) + len(txt)

    # Guarantee something
    if not out_parts:
        out_parts = [scored[0][2][:max_chars]]

    return "".join(out_parts)


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
