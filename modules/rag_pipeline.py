
import ollama

from modules.embeddings import generate_embeddings
from modules.retriever import search_similar_chunks


def generate_rag_answer(question):
    """
    Generate an answer using retrieved document context.
    """

    # Step 1: Convert question into an embedding
    question_embedding = generate_embeddings([question])[0]

    # Step 2: Retrieve relevant document chunks
    results = search_similar_chunks(question_embedding, top_k=2)

    retrieved_chunks = results["documents"][0]

    # Step 3: Combine retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)

    # Step 4: Create a prompt for TinyLlama
    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the information
    provided in the context below.

    If the answer is not available in the context,
    clearly say that the information is not available.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Step 5: Generate answer using TinyLlama
    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
