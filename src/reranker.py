from sentence_transformers import CrossEncoder

from hybrid_retrieval import hybrid_search


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = CrossEncoder(MODEL_NAME)

def rerank(query, candidates, top_k=5):

    # Load cross-encoder
   
    # Create query-document pairs
    pairs = []

    for candidate in candidates:

        pairs.append([
            query,
            candidate["text"]
        ])

    # Calculate relevance scores
    scores = model.predict(pairs)

    # Attach scores
    for candidate, score in zip(
        candidates,
        scores
    ):

        candidate["rerank_score"] = float(score)

    # Sort by reranking score
    ranked_results = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_results[:top_k]


if __name__ == "__main__":

    query = "What datasets were used in this research?"

    # First retrieve candidates using hybrid retrieval
    candidates = hybrid_search(
        query,
        top_k=10
    )

    print("\nQUERY:")
    print(query)

    print("\nRERANKED RESULTS:")
    print("=" * 80)

    results = rerank(
        query,
        candidates,
        top_k=5
    )

    for i, result in enumerate(results):

        metadata = result["metadata"]

        print(f"\nResult {i + 1}")
        print("-" * 80)

        print(
            "Rerank Score:",
            result["rerank_score"]
        )

        print(
            "RRF Score:",
            result["rrf_score"]
        )

        print(
            "Source:",
            metadata["source"]
        )

        print(
            "Page:",
            metadata["page"]
        )

        print(
            "Chunk ID:",
            metadata["chunk_id"]
        )

        print("\nText:")
        print(result["text"][:1500])