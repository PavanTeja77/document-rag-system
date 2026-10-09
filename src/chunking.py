from ingestion import load_pdf


def create_chunks(pages, chunk_size=800, overlap=150):
    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        words = text.split()

        start = 0
        chunk_number = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{chunk_number}"
            })

            chunk_number += 1

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    pdf_path = "data/papers/paper1.pdf"

    pages = load_pdf(pdf_path)

    chunks = create_chunks(pages)

    print(f"Number of pages: {len(pages)}")
    print(f"Number of chunks: {len(chunks)}")

    print("\nFirst chunk:")
    print(chunks[0]["text"])

    print("\nMetadata:")
    print(chunks[0])