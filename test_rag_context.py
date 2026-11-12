
from modules.document_processor import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from modules.embeddings import generate_embeddings
from modules.retriever import store_embeddings
from modules.rag_pipeline import generate_rag_answer_with_context


# Step 1: Load the PDF
pdf_path = "documents/sample.pdf"
text = extract_text_from_pdf(pdf_path)

# Step 2: Split the PDF into chunks
chunks = split_text_into_chunks(text)

# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 4: Store chunks in ChromaDB
store_embeddings(chunks, embeddings)

# Step 5: Ask a question
question = "What is machine learning?"

# Step 6: Generate answer and retrieve contexts
result = generate_rag_answer_with_context(question)

# Step 7: Display the answer
print("\nGENERATED ANSWER:")
print(result["answer"])

# Step 8: Display retrieved contexts
print("\nRETRIEVED CONTEXTS:")

for i, context in enumerate(result["retrieved_contexts"]):
    print(f"\nContext {i + 1}:")
    print(context)

# Step 9: Display available fields
print("\nAVAILABLE FIELDS:")
print(result.keys())
