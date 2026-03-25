from src.embedder import get_embedding


def retrieve_relevant_chunks(query: str, vector_store, top_k: int = 3):
    query_embedding = get_embedding(query)
    return vector_store.search(query_embedding, top_k=top_k)