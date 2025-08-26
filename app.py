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

ARTIFACT_FIXES = (
    ("ΓÇô", "–"), ("â€“", "–"), ("â€”", "—"),
    ("\u00A0", " "), ("\t", " "),
)

GLUE_PATTERNS = [
    # “75, 000” -> “75,000”
    (re.compile(r"(\d),\s+(\d{3})"), r"\1,\2"),
    # ensure space after sentence enders when followed by a letter/number
    (re.compile(r"([.!?])([A-Za-z0-9])"), r"\1 \2"),
    # fix mid-paragraph glued words when capitalization suggests a boundary
    (re.compile(r"([a-z])([A-Z])"), r"\1 \2"),
]

def normalize_text(s: str) -> str:
    if not s:
        return s
    for bad, good in ARTIFACT_FIXES:
        s = s.replace(bad, good)
    # collapse multi-spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    # line level fixes
    lines = []
    for ln in s.splitlines():
        # drop bare page numbers (common extractor noise)
        if re.fullmatch(r"\s*\d+\s*", ln):
            continue
        for pat, repl in GLUE_PATTERNS:
            ln = pat.sub(repl, ln)
        lines.append(ln.strip())
    s = "\n".join(lines)
    return s.strip()


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
    if any(k in name for k in ["loan", "promissory"]) or "loan agreement" in t:
        return "Contract"
    if any(k in name for k in ["invoice", "receipt"]) or any(k in t for k in ["invoice #", "amount due"]):
        return "Invoice"
    if any(k in name for k in ["resume", "cv"]) or any(k in t for k in ["education", "experience", "skills"]):
        return "Resume"
    if "non-disclosure" in name or "nda" in t:
        return "Contract"
    if "report" in name:
        return "Report"
    return "Unknown"


# ---- summarization ------------------------------------------------------------
SECTION_KEYWORDS = {
    "parties": {"party", "parties", "lender", "borrower", "company", "entity", "address", "principal office", "disclosing", "receiving"},
    "purpose": {"purpose", "objective", "consideration", "loan", "collaboration", "evaluation"},
    "loan_terms": {"loan", "principal", "amount", "interest", "rate", "percentage", "apr", "term", "duration", "effective", "expiration", "terminate", "termination"},
    "repayment": {"repayment", "installment", "schedule", "payment", "due", "monthly", "amortization"},
    "obligations": {"obligation", "duty", "maintain", "protect", "insurance", "covenant", "collateral", "agree"},
    "default": {"default", "breach", "late", "remedy", "cure", "acceleration", "repossession"},
    "governing": {"governing", "law", "jurisdiction", "venue", "state", "county", "country"},
    "signatures": {"signature", "execute", "date", "witness", "notary", "agreement"}
}

def _sentences(text: str) -> List[str]:
    if SPACY_OK and nlp:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # tiny fallback: split on period/end punctuation
    raw = re.split(r'(?<=[\.!?])\s+', text)
    return [r.strip() for r in raw if r.strip()]

def summarize(text: str, max_sentences: int = 15, max_chars: int = 2600) -> str:
    """
    Extractive summary tuned for contracts/loans.
    Tries to cover key sections, with length raised for professional use.
    """
    if not text:
        return ""

    try:
        doc = nlp(text)
    except Exception:
        return text[:max_chars]

    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return text[:max_chars]

    # lemma frequency
    freqs: dict[str, float] = {}
    for t in doc:
        if t.is_stop or t.is_punct or t.like_num or t.is_space:
            continue
        lem = t.lemma_.lower().strip()
        if lem:
            freqs[lem] = freqs.get(lem, 0.0) + 1.0
    if not freqs:
        return " ".join(s.text.strip() for s in sents[:max_sentences])[:max_chars]

    max_f = max(freqs.values())
    for k in list(freqs):
        freqs[k] /= max_f

    scored = []  # (idx, score, text, sections_hit:set)
    for i, s in enumerate(sents):
        lemmas = [t.lemma_.lower() for t in s if not (t.is_stop or t.is_punct or t.like_num or t.is_space)]
        if not lemmas:
            continue
        base = sum(freqs.get(l, 0.0) for l in lemmas)
        length_norm = max(1.0, len(lemmas) ** 0.85)

        secs = set()
        sl = set(lemmas)
        for sec, kws in SECTION_KEYWORDS.items():
            if sl.intersection(kws):
                secs.add(sec)

        # heavier boost for especially important sections
        boost_map = {
            "loan_terms": 1.6, "repayment": 1.6, "default": 1.5,
            "parties": 1.4, "governing": 1.3, "obligations": 1.3,
            "purpose": 1.2, "signatures": 1.1
        }
        boost = max([1.0] + [boost_map.get(sec, 1.0) for sec in secs])
        score = (base / length_norm) * boost
        scored.append((i, score, s.text.strip(), secs))

    if not scored:
        return " ".join(s.text.strip() for s in sents[:max_sentences])[:max_chars]

    scored.sort(key=lambda x: x[1], reverse=True)

    # ensure coverage: pick best per priority section first
    priority = ["loan_terms", "repayment", "default", "parties", "obligations", "governing", "purpose", "signatures"]
    chosen, used = [], set()
    for sec in priority:
        for tup in scored:
            i, _, txt, secs = tup
            if i in used:
                continue
            if sec in secs:
                chosen.append(tup); used.add(i); break
        if len(chosen) >= max_sentences:
            break

    # fill remaining by score
    for tup in scored:
        if len(chosen) >= max_sentences:
            break
        i = tup[0]
        if i not in used:
            chosen.append(tup); used.add(i)

    # back to original order, pack within char budget
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


def extract_structured_terms(text: str) -> list[dict]:
    """
    Regex helpers for loan/contract essentials; complements spaCy entities.
    Produces items with the same {label, text} shape your Java expects.
    """
    items = []
    # interest rate (e.g., 6.5%, 5 % per annum)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:per\s+annum|annual|apr)?", text, flags=re.I)
    if m: items.append({"label": "INTEREST_RATE", "text": m.group(0)})

    # monthly installment (currency + amount)
    m = re.search(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(?:per\s+month|monthly|installment)", text, flags=re.I)
    if m: items.append({"label": "MONTHLY_PAYMENT", "text": m.group(0)})

    # principal amount
    m = re.search(r"(?:principal\s+amount|loan\s+amount)\s+of\s+\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text, flags=re.I)
    if m: items.append({"label": "PRINCIPAL", "text": m.group(0)})

    # term length (e.g., 60 monthly installments)
    m = re.search(r"(\d+)\s+(?:monthly|months)\s+(?:installments|payments?)", text, flags=re.I)
    if m: items.append({"label": "TERM", "text": m.group(0)})

    # effective date
    m = re.search(r"(effective\s+date[:\s]+[A-Z][a-z]+\s+\d{1,2},\s+\d{4})", text, flags=re.I)
    if m: items.append({"label": "EFFECTIVE_DATE", "text": m.group(1)})

    # governing law
    m = re.search(r"laws?\s+of\s+the\s+state\s+of\s+([A-Za-z ]+)", text, flags=re.I)
    if m: items.append({"label": "GOVERNING_LAW", "text": m.group(0)})

    # party names (very light heuristic; spaCy PERSON/ORG will also catch most)
    m = re.search(r"lender[:\s]+(.+?)[\.,\n]", text, flags=re.I)
    if m: items.append({"label": "LENDER", "text": m.group(1).strip()})
    m = re.search(r"borrower[:\s]+(.+?)[\.,\n]", text, flags=re.I)
    if m: items.append({"label": "BORROWER", "text": m.group(1).strip()})

    return items


# ---- entity extraction ---------------------------------------------------------
def extract_fields(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    results: List[Dict[str, Any]] = []
    try:
        # spaCy entities (when available)
        if SPACY_OK and nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY"}:
                    results.append({"label": ent.label_, "text": ent.text})
        else:
            # lightweight fallback: dates + money
            for m in re.finditer(r"\b(?:[A-Z][a-z]+\s\d{1,2},\s\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b", text):
                results.append({"label": "DATE", "text": m.group(0)})
            for m in re.finditer(r"\$\s?\d[\d,]*(?:\.\d{2})?", text):
                results.append({"label": "MONEY", "text": m.group(0)})

        # always add structured finance/legal terms
        results.extend(extract_structured_terms(text))

        # de-duplicate while preserving order
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in results:
            key = (item.get("label", ""), re.sub(r"\s+", " ", item.get("text", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"label": key[0], "text": key[1]})

        # allow a bit more room now that we add structured items
        return deduped[:80]

    except Exception:
        # ultra-safe fallback if anything above errors
        safe: List[Dict[str, Any]] = []
        for m in re.finditer(r"\b(?:[A-Z][a-z]+\s\d{1,2},\s\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b", text):
            safe.append({"label": "DATE", "text": m.group(0)})
        for m in re.finditer(r"\$\s?\d[\d,]*(?:\.\d{2})?", text):
            safe.append({"label": "MONEY", "text": m.group(0)})
        # still try to add structured terms; guard with try just in case
        try:
            safe.extend(extract_structured_terms(text))
        except Exception:
            pass
        return safe[:80]


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

        text = normalize_text(text)
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
