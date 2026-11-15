
import re
import ollama

from modules.embeddings import generate_embeddings
from modules.retriever import search_similar_chunks


def has_keyword_overlap(question, context):
    """
    Check whether important question keywords
    appear in the retrieved document context.
    """

    stopwords = {
        "what", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on",
        "to", "for", "and", "or", "how",
        "does", "do", "can", "could", "explain",
        "describe", "define", "about", "with",
        "tell", "me", "give", "show"
    }

    question_words = set(
        re.findall(r"\b[a-zA-Z]+\b", question.lower())
    )

    context_words = set(
        re.findall(r"\b[a-zA-Z]+\b", context.lower())
    )

    important_words = question_words - stopwords

    if not important_words:
        return True

    matching_words = important_words.intersection(
        context_words
    )

    overlap_ratio = (
        len(matching_words) / len(important_words)
    )

    return overlap_ratio >= 0.5


def generate_rag_answer_with_context(question, document_id):
    """
    Generate an answer using the selected PDF
    and return the retrieved contexts.
    """

    # Step 1: Generate question embedding
    question_embedding = generate_embeddings([question])[0]

    # Step 2: Retrieve chunks from selected PDF
    results = search_similar_chunks(
        question_embedding,
        document_id=document_id,
        top_k=2
    )

    # Step 3: Extract retrieved information
    retrieved_chunks = (
        results["documents"][0]
        if results.get("documents") and results["documents"][0]
        else []
    )

    distances = (
        results["distances"][0]
        if results.get("distances") and results["distances"][0]
        else []
    )

    if not retrieved_chunks or not distances:

        return {
            "answer": (
                "No relevant information was found "
                "in the selected document."
            ),
            "retrieved_contexts": []
        }

    print("\nDEBUG - Retrieved distances:")
    print(distances)

    # Step 4: Check retrieval distance
    DISTANCE_THRESHOLD = 1.5

    if min(distances) > DISTANCE_THRESHOLD:

        return {
            "answer": (
                "The information is not available "
                "in the provided document."
            ),
            "retrieved_contexts": []
        }

    # Step 5: Combine retrieved chunks
    context = "\n\n".join(retrieved_chunks)

    # Step 6: Check keyword relevance
    keyword_check = has_keyword_overlap(
        question,
        context
    )

    print("\nDEBUG - Question:", question)
    print("DEBUG - Keyword check:", keyword_check)

    if not keyword_check:

        print("DEBUG - Question rejected by keyword check")

        return {
            "answer": (
                "The information is not available "
                "in the provided document."
            ),
            "retrieved_contexts": []
        }

    print("DEBUG - Question passed keyword check")

    # Step 7: Create prompt for TinyLlama
    prompt = f"""
You are a question-answering assistant.

Use the context below to answer the question.
Do not repeat these instructions.
Do not invent information.
Do not include information from outside the context.
Answer only the question asked.
Use simple language and short sentences.
If the answer is missing, say:
"The information is not available in the provided document."

Context:
{context}

Question:
{question}

Give the answer now:
"""

    # Step 8: Generate answer
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

    # Step 9: Return result
    return {
        "answer": answer,
        "retrieved_contexts": retrieved_chunks
    }


def generate_rag_answer(question, document_id):
    """
    Return only the generated answer.
    """

    result = generate_rag_answer_with_context(
        question,
        document_id
    )

    return result["answer"]
