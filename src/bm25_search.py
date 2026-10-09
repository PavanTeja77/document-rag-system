import pickle


BM25_PATH = "vectorstore/bm25.pkl"


def bm25_search(query, top_k=5):

    # Load the saved BM25 index
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    # Tokenize the user's query
    query_tokens = query.lower().split()

    # Calculate BM25 score for every chunk
    scores = bm25.get_scores(query_tokens)

    # Rank chunks from highest score to lowest score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "chunk": chunks[index],
            "score": scores[index]
        })

    return results


if __name__ == "__main__":

    query = "What datasets were used in this research?"

    results = bm25_search(query, top_k=5)

    print("\nQUERY:")
    print(query)

    print("\nBM25 RESULTS:")
    print("=" * 80)

    for i, result in enumerate(results):

        chunk = result["chunk"]

        print(f"\nResult {i + 1}")
        print("-" * 80)

        print("Score:", result["score"])
        print("Source:", chunk["source"])
        print("Page:", chunk["page"])
        print("Chunk ID:", chunk["chunk_id"])

        print("\nText:")
        print(chunk["text"][:1500])