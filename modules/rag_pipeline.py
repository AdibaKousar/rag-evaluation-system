
import ollama

from modules.embeddings import generate_embeddings
from modules.retriever import search_similar_chunks


def generate_rag_answer_with_context(question):
    """
    Generate an answer and return the retrieved contexts.
    """

    # Step 1: Convert question into an embedding
    question_embedding = generate_embeddings([question])[0]

    # Step 2: Retrieve relevant document chunks
    results = search_similar_chunks(question_embedding, top_k=2)

    retrieved_chunks = results["documents"][0]

    # Step 3: Combine chunks into context
    context = "\n\n".join(retrieved_chunks)

    # Step 4: Create prompt for TinyLlama
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

    answer = response["message"]["content"]

    # Step 6: Return answer and retrieved contexts
    return {
        "answer": answer,
        "retrieved_contexts": retrieved_chunks
    }


def generate_rag_answer(question):
    """
    Return only the generated answer.
    Keeps compatibility with the existing RAG test.
    """

    result = generate_rag_answer_with_context(question)

    return result["answer"]
