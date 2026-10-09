import re


# Keep this False initially.
# BM25 already uses IDF to reduce the importance of common words.
REMOVE_STOPWORDS = False


STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "if", "then", "than", "of", "to", "in",
    "on", "for", "from", "by", "with",
    "is", "are", "was", "were", "be",
    "been", "being", "this", "that",
    "these", "those", "it", "its",
    "as", "at", "into", "about",
    "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "can", "could", "should", "would",
    "will", "shall", "do", "does",
    "did", "used", "using"
}


def tokenize(text):

    # Convert to lowercase
    text = text.lower()

    # Extract meaningful word/number tokens
    tokens = re.findall(
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*",
        text
    )

    # Optionally remove stopwords
    if REMOVE_STOPWORDS:
        tokens = [
            token
            for token in tokens
            if token not in STOPWORDS
        ]

    return tokens