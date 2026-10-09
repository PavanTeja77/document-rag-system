from retrieval import retrieve
from bm25_search import bm25_search


RRF_K = 60


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=5
):

    scores = {}
    documents = {}

    # --------------------------------------------------
    # 1. Process dense retrieval results
    # --------------------------------------------------

    dense_documents = dense_results["documents"][0]
    dense_metadatas = dense_results["metadatas"][0]

    for rank, (document, metadata) in enumerate(
        zip(dense_documents, dense_metadatas),
        start=1
    ):

        chunk_id = metadata["chunk_id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        documents[chunk_id] = {
            "text": document,
            "metadata": metadata
        }

    # --------------------------------------------------
    # 2. Process BM25 results
    # --------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk = result["chunk"]

        chunk_id = chunk["chunk_id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        documents[chunk_id] = {
            "text": chunk["text"],
            "metadata": {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
        }

    # --------------------------------------------------
    # 3. Sort by RRF score
    # --------------------------------------------------

    ranked_chunks = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------
    # 4. Return top-k results
    # --------------------------------------------------

    results = []

    for chunk_id, score in ranked_chunks[:top_k]:

        result = documents[chunk_id].copy()

        result["rrf_score"] = score

        results.append(result)

    return results


def hybrid_search(query, top_k=5):

    # Retrieve candidates from dense retrieval
    dense_results = retrieve(
        query,
        top_k=10
    )

    # Retrieve candidates from BM25
    bm25_results = bm25_search(
        query,
        top_k=10
    )

    # Fuse both rankings
    results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=top_k
    )

    return results


if __name__ == "__main__":

    query = "What datasets were used in this research?"

    results = hybrid_search(
        query,
        top_k=10
    )

    print("\nQUERY:")
    print(query)

    print("\nHYBRID RESULTS:")
    print("=" * 80)

    for i, result in enumerate(results):

        metadata = result["metadata"]

        print(f"\nResult {i + 1}")
        print("-" * 80)

        print("RRF Score:", result["rrf_score"])
        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Chunk ID:", metadata["chunk_id"])

        print("\nText:")
        print(result["text"][:1500])