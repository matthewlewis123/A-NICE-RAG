# --- Imports
import json
import string
from typing import Dict, List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

for resource in ["punkt", "stopwords", "wordnet"]:
    try:
        resource_path = {
            "punkt": "tokenizers/punkt",
            "stopwords": "corpora/stopwords",
            "wordnet": "corpora/wordnet",
        }
        nltk.data.find(resource_path[resource])
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource)


def load_chunked_data(file_path: str) -> List[Dict]:
    """Load chunked data from JSON file."""

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_text(text: str, use_lemmatization: bool = False) -> List[str]:
    """Preprocess the input text by tokenizing, removing stopwords, and optionally lemmatizing."""

    if not text:
        return []

    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stopwords_set = set(stopwords.words("english"))
    tokens = [
        token
        for token in tokens
        if token not in stopwords_set and not token.isnumeric() and len(token) > 1
    ]

    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def preprocess_chunks_to_csv():
    """Save preprocessed chunks to a CSV file."""

    chunks = load_chunked_data("data/chunked_guidelines.json")
    data = []

    with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
        for i, chunk in enumerate(chunks):
            if not chunk.get("title") or not chunk.get("content"):
                pbar.update(1)
                continue

            title = chunk["title"]
            content = chunk.get("content", "")
            source = chunk.get("source", "unknown")

            tokens_regular = preprocess_text(content, use_lemmatization=False)
            tokens_lemmatized = preprocess_text(content, use_lemmatization=True)

            data.append(
                {
                    "id": title,
                    "source": source,
                    "content": content,
                    "tokens_regular": tokens_regular,
                    "tokens_lemmatized": tokens_lemmatized,
                }
            )

            pbar.update(1)

    print(f"Preprocessing complete: {len(data)} valid chunks processed")

    df = pd.DataFrame(data)

    output_file = "data/chunks_bm25.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved {len(df)} chunks to {output_file}")

    return df


if __name__ == "__main__":
    df = preprocess_chunks_to_csv()
