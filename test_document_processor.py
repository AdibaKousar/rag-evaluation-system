from modules.document_processor import (
    extract_text_from_pdf,
    split_text_into_chunks
)

pdf_path = "documents/sample.pdf"

# Extract text
text = extract_text_from_pdf(pdf_path)

print("EXTRACTED TEXT:")
print(text)

# Split text into chunks
chunks = split_text_into_chunks(text)

print("\nTOTAL CHUNKS:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nCHUNK {i + 1}:")
    print(chunk)