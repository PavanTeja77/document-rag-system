import chromadb

from ingestion import load_pdf
from chunking import create_chunks
from embeddings import create_embeddings


PDF_PATH = "data/papers/paper1.pdf"
DB_PATH = "vectorstore"
COLLECTION_NAME = "research_papers"


def create_vector_store():

    # Step 1: Load PDF
    pages = load_pdf(PDF_PATH)

    # Step 2: Create chunks
    chunks = create_chunks(pages)

    # Step 3: Generate embeddings
    embeddings = create_embeddings(chunks)

    # Step 4: Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    # Step 5: Create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Step 6: Store chunks, embeddings and metadata
    collection.upsert(
        ids=[
            chunk["chunk_id"]
            for chunk in chunks
        ],

        documents=[
            chunk["text"]
            for chunk in chunks
        ],

        embeddings=embeddings.tolist(),

        metadatas=[
    {
        "source": chunk["source"],
        "page": chunk["page"],
        "chunk_id": chunk["chunk_id"]
    }
    for chunk in chunks
]
    )

    print("\nVector database created successfully!")

    print(
        "Number of stored chunks:",
        collection.count()
    )


if __name__ == "__main__":
    create_vector_store()