import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add_embeddings(self, embeddings: list[list[float]], chunks: list[str]) -> None:
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(dist)))

        return results