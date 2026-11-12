
from modules.document_processor import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from modules.embeddings import generate_embeddings

from modules.retriever import store_embeddings

from modules.rag_pipeline import generate_rag_answer


# Step 1: Load the PDF
pdf_path = "documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)


# Step 2: Split the PDF text
chunks = split_text_into_chunks(text)


# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)


# Step 4: Store the chunks in ChromaDB
store_embeddings(chunks, embeddings)


# Step 5: Ask a question
question = input("\nEnter your question: ")


# Step 6: Generate the RAG answer
answer = generate_rag_answer(question)


# Step 7: Display the answer
print("\nYOUR RAG ANSWER:")
print(answer)
