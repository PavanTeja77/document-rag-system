import pickle

from rank_bm25 import BM25Okapi

from ingestion import load_pdf
from chunking import create_chunks


PDF_PATH = "data/papers/paper1.pdf"
BM25_PATH = "vectorstore/bm25.pkl"


def build_bm25_index():

    pages = load_pdf(PDF_PATH)

    chunks = create_chunks(pages)

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks
            },
            f
        )

    print("BM25 index created successfully.")
    print("Number of indexed chunks:", len(chunks))


if __name__ == "__main__":
    build_bm25_index()