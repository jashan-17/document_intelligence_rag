def retrieve_relevant_chunks(query: str, vector_store, top_k: int = 3):
    return vector_store.search(query, top_k=top_k)
