from reranker import rerank
from hybrid_retrieval import hybrid_search


def build_context(query, top_k=3):

    # Get hybrid candidates
    candidates = hybrid_search(
        query,
        top_k=10
    )

    # Rerank candidates
    results = rerank(
        query,
        candidates,
        top_k=top_k
    )

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source = metadata["source"]
        page = metadata["page"]
        chunk_id = metadata["chunk_id"]

        text = result["text"]

        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{text}"
        )

    context = "\n\n".join(context_parts)

    return context, results


if __name__ == "__main__":

    query = "What datasets were used in this research?"

    context, results = build_context(
        query,
        top_k=5
    )

    print("\nQUERY:")
    print(query)

    print("\nCONTEXT:")
    print("=" * 80)

    print(context)