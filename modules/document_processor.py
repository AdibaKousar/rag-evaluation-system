
import re

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_extracted_text(text):
    """Remove unwanted PDF footer metadata."""

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove local file paths
        if "file:///" in line.lower():
            continue

        # Remove common PDF page footer metadata
        if re.search(r"\d{1,2}/\d{1,2}/\d{4}", line):
            continue

        if line.lower().endswith(".pdf"):
            continue

        if line.lower().endswith(".txt"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_text_from_pdf(pdf_path):
    """Extract and clean text from a PDF file."""

    reader = PdfReader(pdf_path)

    extracted_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            extracted_text += text + "\n"

    # Clean unwanted metadata
    extracted_text = clean_extracted_text(extracted_text)

    return extracted_text


def split_text_into_chunks(text):
    """Divide extracted text into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    return chunks
