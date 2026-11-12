
import chromadb


# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")


# Create or access a collection
collection = client.get_or_create_collection(
    name="rag_documents"
)


def store_embeddings(chunks, embeddings):
    """
    Store text chunks and their embeddings in ChromaDB.
    """

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )

    print("Embeddings stored successfully!")


def search_similar_chunks(query_embedding, top_k=2):
    """
    Search for the most relevant chunks.
    """

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results
