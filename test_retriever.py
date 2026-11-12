
from modules.document_processor import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from modules.embeddings import generate_embeddings

from modules.retriever import (
    store_embeddings,
    search_similar_chunks
)


# Step 1: Read the PDF
pdf_path = "documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)


# Step 2: Split text into chunks
chunks = split_text_into_chunks(text)


# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)


# Step 4: Store chunks and embeddings
store_embeddings(chunks, embeddings)


# Step 5: Ask a question
question = "What is machine learning?"

question_embedding = generate_embeddings([question])[0]


# Step 6: Search for relevant chunks
results = search_similar_chunks(question_embedding)


# Step 7: Display results
print("\nSEARCH RESULTS:")

for i, result in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(result)
