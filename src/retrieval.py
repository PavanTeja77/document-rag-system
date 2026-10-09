import chromadb

from sentence_transformers import SentenceTransformer


DB_PATH = "vectorstore"
COLLECTION_NAME = "research_papers"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def retrieve(query, top_k=5):

    # Load embedding model
   

    # Connect to existing vector database
    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    # Get our collection
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Convert user query into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Search vector database
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    query = "What datasets were used in this research?"

    results = retrieve(query, top_k=5)

    print("\nQUERY:")
    print(query)

    print("\nRETRIEVED CHUNKS:")
    print("=" * 80)

    for i in range(len(results["documents"][0])):

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nResult {i + 1}")
        print("-" * 80)

        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Distance:", distance)

        print("\nText:")
        print(document[:1500])