# --- Imports --
import pandas as pd
from preprocess_queries import preprocess_query
from tqdm import tqdm


def extract_queries_from_csv(csv_path: str) -> pd.DataFrame:
    """Extract queries from suggested_queries.csv"""

    df = pd.read_csv(csv_path)
    queries_df = pd.DataFrame({"id": df["id"], "query": df["query"]})

    return queries_df.dropna(subset=["query"])


def preprocess_suggested_queries_for_bm25(csv_path: str, output_path: str):
    """Preprocess suggested queries for BM25 optimization"""

    df = pd.read_csv(csv_path)
    queries_df = pd.DataFrame({"id": df["id"], "query": df["query"]}).dropna(subset=["query"])

    preprocessed_data = []

    print(f"Preprocessing {len(queries_df)} queries")
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Processing queries"):
        query = row["query"]
        query_id = row["id"]

        tokens_regular = preprocess_query(query, use_lemmatization=False)
        tokens_lemmatized = preprocess_query(query, use_lemmatization=True)

        preprocessed_data.append(
            {
                "id": query_id,
                "query": query,
                "tokens_regular": tokens_regular,
                "tokens_lemmatized": tokens_lemmatized,
            }
        )

    result_df = pd.DataFrame(preprocessed_data)
    result_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(result_df)} preprocessed queries to {output_path}")
    return result_df


if __name__ == "__main__":
    queries_df = extract_queries_from_csv("databases/suggested_queries.csv")
    print(f"Extracted {len(queries_df)} queries from CSV")
    preprocess_suggested_queries_for_bm25(
        "../../databases/suggested_queries.csv",
        "../../data/suggested_queries_bm25_preprocessed.csv",
    )
