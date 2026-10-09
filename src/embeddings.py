from sentence_transformers import SentenceTransformer
from ingestion import load_pdf
from chunking import create_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":

    pdf_path = "data/papers/paper1.pdf"

    pages = load_pdf(pdf_path)

    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    print("\nEmbedding complete.")

    print("Number of chunks:", len(chunks))

    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])