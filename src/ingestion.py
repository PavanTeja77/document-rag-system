import pymupdf as fitz
from pathlib import Path


def load_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number + 1,
            "source": Path(pdf_path).name
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/papers/paper1.pdf"

    pages = load_pdf(pdf_path)

    print(f"Number of pages: {len(pages)}")

    print("\nFirst page:")
    print(pages[0]["text"][:2000])