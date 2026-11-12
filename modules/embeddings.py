
from sentence_transformers import SentenceTransformer


# Load the free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(text_chunks):
    """
    Convert text chunks into numerical vectors.
    """

    embeddings = model.encode(text_chunks)

    return embeddings
