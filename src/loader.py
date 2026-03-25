import fitz 
from docx import Document
from pathlib import Path


#Opens the PDF, reads text from each page and joins them into a big string
def load_pdf(file_path: str) -> str:
    text = []
    doc = fitz.open(file_path)
    for page in doc:
        text.append(page.get_text())
    doc.close()
    return "\n".join(text)

#Opens the word file, reads all paragraphs and joins them into one string 
def load_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

#Checks file extension to decide to use the PDF or DOCX loader
def load_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")