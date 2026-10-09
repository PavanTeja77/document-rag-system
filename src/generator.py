import ollama

from context_builder import build_context


MODEL_NAME = "qwen2.5:3b"


def generate_answer(query):

    # Retrieve and rerank
    context, sources = build_context(
        query,
        top_k=3
    )

    prompt = f"""
You are a question-answering system for a research paper.

Rules:
- Answer ONLY from the provided context.
- Do not use outside knowledge.
- If the answer is not present in the context, say:
  "The answer was not found in the provided document."
- Be concise and factual.
- Do NOT generate citations or page numbers yourself.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    answer = response["message"]["content"]

    return answer, sources

def add_citations(answer, sources):
    if not sources:
        return answer

    abstention_phrases = [
        "answer was not found",
        "not found in the provided document",
        "information is not available",
        "not mentioned in the document"
    ]

    if any(phrase in answer.lower() for phrase in abstention_phrases):
        return answer

    best_source = sources[0]["metadata"]

    citation = (
        f"[{best_source['source']}, "
        f"Page {best_source['page']}]"
    )

    return answer + "\n\nSource: " + citation

if __name__ == "__main__":

    query = "did it mention resampling?"

    answer, sources = generate_answer(query)

    final_answer = add_citations(
        answer,
        sources
    )

    print("\nQUESTION:")
    print(query)

    print("\nANSWER:")
    print("=" * 80)

    print(final_answer)