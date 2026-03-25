import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]], model: str = "gpt-4.1-mini") -> str:
    context = "\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are a document question-answering assistant.
Answer the question using only the provided context.
If the answer is not in the context, say you could not find it in the documents.

Context:
{context}

Question:
{query}

Give a concise answer and cite the source numbers used.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You answer questions only from the provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content