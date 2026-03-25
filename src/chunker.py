from typing import List
#Takes full document and splits into chunks

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks