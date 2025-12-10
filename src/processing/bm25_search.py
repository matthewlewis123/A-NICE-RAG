# --- Imports
import ast
import os
import pickle

import nltk
import pandas as pd
from langchain.schema.document import Document
from rank_bm25 import BM25Okapi
from tqdm import tqdm

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def load_preprocessed_chunks(csv_path: str) -> pd.DataFrame:
    """Load preprocessed chunks from CSV"""

    df = pd.read_csv(csv_path, encoding="utf-8")

    def safe_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return []

    df["tokens_regular"] = df["tokens_regular"].apply(safe_eval)
    df["tokens_lemmatized"] = df["tokens_lemmatized"].apply(safe_eval)

    return df


def index_with_bm25(
    chunks_df: pd.DataFrame,
    use_lemmatized: bool = True,
    k1: float = 1.7,
    b: float = 0.83,
    epsilon: float = 0.05,
):
    """Index chunks with BM25 using preprocessed tokens."""

    token_column = "tokens_lemmatized" if use_lemmatized else "tokens_regular"

    sections = []
    section_ids = []
    corpus = []

    print("Creating Document objects and corpus...")
    for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Processing chunks"):
        chunk_id = row["id"]
        source = row["source"]
        content = row["content"]
        tokens = row[token_column]

        if not tokens or len(tokens) == 0:
            continue

        doc = Document(page_content=content, metadata={"id": chunk_id, "source": source})
        sections.append(doc)
        section_ids.append(chunk_id)
        corpus.append(tokens)

    print(f"Building BM25 index with {len(corpus)} documents...")
    print(f"Parameters: k1={k1}, b={b}, epsilon={epsilon}")
    bm25 = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)

    return bm25, sections, section_ids


def export_bm25_to_file(bm25, sections, section_ids, filepath, config_info=None):
    """Export BM25 index to pickle file with configuration info."""
    data = {
        "bm25": bm25,
        "sections": sections,
        "section_ids": section_ids,
        "config": config_info or {},
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def main(
    preprocessed_csv_path: str,
    output_pickle_path: str,
    use_lemmatized: bool = True,
    k1: float = 1.7,
    b: float = 0.83,
    epsilon: float = 0.05,
):
    """Create BM25 index from preprocessed CSV data."""

    chunks_df = load_preprocessed_chunks(preprocessed_csv_path)
    print(f"Loaded {len(chunks_df)} preprocessed chunks")

    print("Indexing with BM25...")
    bm25, sections, section_ids = index_with_bm25(
        chunks_df, use_lemmatized=use_lemmatized, k1=k1, b=b, epsilon=epsilon
    )
    print(f"Created BM25 index with {len(sections)} sections")

    config_info = {
        "use_lemmatized": use_lemmatized,
        "tokenization_type": "lemmatized" if use_lemmatized else "regular",
        "k1": k1,
        "b": b,
        "epsilon": epsilon,
        "total_documents": len(sections),
    }

    print(f"Exporting BM25 index to {output_pickle_path}")
    export_bm25_to_file(
        bm25, sections, section_ids, filepath=output_pickle_path, config_info=config_info
    )

    print("BM25 index creation complete")

    return bm25, sections, section_ids


if __name__ == "__main__":
    PREPROCESSED_CSV_PATH = "data/chunks_bm25.csv"
    USE_LEMMATIZED = True
    K1 = 1.7
    B = 0.83
    EPSILON = 0.05

    OUTPUT_PICKLE_PATH = f"databases/bm25_index_nice_guidelines.pkl"

    main(
        PREPROCESSED_CSV_PATH,
        OUTPUT_PICKLE_PATH,
        use_lemmatized=USE_LEMMATIZED,
        k1=K1,
        b=B,
        epsilon=EPSILON,
    )
