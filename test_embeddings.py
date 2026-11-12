
from modules.document_processor import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from modules.embeddings import generate_embeddings


# Read the PDF
pdf_path = "documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)

# Divide text into chunks
chunks = split_text_into_chunks(text)

# Generate embeddings
embeddings = generate_embeddings(chunks)

# Display results
print("TOTAL TEXT CHUNKS:", len(chunks))

print("\nEMBEDDING SHAPE:", embeddings.shape)

print("\nFIRST EMBEDDING:")
print(embeddings[0])

print("\nFIRST 5 VALUES:")
print(embeddings[0][:5])
