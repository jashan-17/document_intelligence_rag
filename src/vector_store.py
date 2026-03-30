from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LocalVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.text_chunks = []
        self.chunk_matrix = None

    def add_documents(self, chunks: list[str]) -> None:
        self.text_chunks = chunks[:]
        self.chunk_matrix = self.vectorizer.fit_transform(self.text_chunks)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.text_chunks or self.chunk_matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                results.append((self.text_chunks[idx], float(scores[idx])))

        if not results and self.text_chunks:
            # Fallback so the app can still show a source even for vague questions.
            results.append((self.text_chunks[0], 0.0))

        return results
