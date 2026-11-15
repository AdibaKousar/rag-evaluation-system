
import chromadb
import uuid


# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")


# Create or access a collection
collection = client.get_or_create_collection(
    name="rag_documents"
)


def store_embeddings(chunks, embeddings, document_name):

    """
    Store document chunks with unique IDs and metadata.
    """

    # Generate a unique ID for this document
    document_id = str(uuid.uuid4())

    # Generate unique chunk IDs
    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    # Store document metadata
    metadatas = [
        {
            "document_id": document_id,
            "document_name": document_name
        }
        for _ in chunks
    ]

    # Store chunks and embeddings
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas
    )

    print("Document stored successfully!")
    print("Document:", document_name)
    print("Total chunks:", len(chunks))

    return document_id


def search_similar_chunks(
    query_embedding,
    document_id,
    top_k=2
):

    """
    Search for relevant chunks from a selected document.
    """

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        where={
            "document_id": document_id
        }
    )

    return results



def get_uploaded_documents():
    results = collection.get(include=["metadatas"])

    documents = {}

    for metadata in results["metadatas"]:

        # Skip older records without metadata
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        document_name = metadata.get("document_name")

        if document_id and document_name:
            documents[document_id] = document_name

    return documents
