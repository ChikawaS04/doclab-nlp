import bytes
import os, io
import str
from flask import Flask, jsonify, request
from flask_cors import CORS
import pdfplumber
import spacy
from docx import Document as DocxDocument

# load spaCy once
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return jsonify(status="ok", service="doclab-nlp", version="0.2.0")

def read_pdf(bytes_data: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_chunks.append(t)
    return "\n".join(text_chunks).strip()

def read_docx(bytes_data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(bytes_data))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def detect_doc_type(text: str, filename: str) -> str:
    name = filename.lower()
    t = text.lower()
    if any(k in name for k in ["invoice", "receipt"]) or any(k in t for k in ["invoice #", "total due", "amount due"]):
        return "Invoice"
    if any(k in name for k in ["resume", "cv"]) or any(k in t for k in ["education", "experience", "skills"]):
        return "Resume"
    if "reference" in name or "references" in t:
        return "Reference"
    if "report" in name:
        return "Report"
    return "Unknown"

def summarize(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return text[:280]
    return " ".join(sents[:max_sentences])[:1000]

def extract_fields(text: str):
    fields = []
    if not text:
        return fields
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY"}:
            fields.append({"fieldName": ent.label_, "fieldValue": ent.text})
    return fields[:50]  # keep it light

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

    text = ""
    try:
        if ext == ".pdf":
            text = read_pdf(raw)
        elif ext == ".docx":
            text = read_docx(raw)
        elif ext in {".txt"}:
            text = raw.decode("utf-8", errors="ignore")
        else:
            # Unknown type for now; OCR comes later with Docker/Tesseract
            return jsonify(error=f"Unsupported file type '{ext}' (PDF/DOCX/TXT only for now)"), 415
    except Exception as e:
        return jsonify(error=f"Failed to parse file: {str(e)}"), 500

    title = (text.splitlines()[0].strip() if text else filename) or filename
    doc_type = detect_doc_type(text, filename)
    summary = summarize(text)
    extracted = extract_fields(text)

    return jsonify(
        docType=doc_type,
        title=title[:200],
        summary=summary,
        extractedFields=extracted
    ), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
