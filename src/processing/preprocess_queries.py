import os
import sqlite3
import string
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        resource_path = {
            "punkt": "tokenizers/punkt",
            "stopwords": "corpora/stopwords",
            "wordnet": "corpora/wordnet",
            "omw-1.4": "corpora/omw-1.4",
        }
        nltk.data.find(resource_path[resource])
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource)


def load_queries(db_path: str) -> pd.DataFrame:
    """Load queries from database"""

    print(f"Loading queries from {db_path}...")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, query FROM queries", conn)
    conn.close()

    print(f"Loaded {len(df)} queries")
    return df


def preprocess_query(query: str, use_lemmatization: bool = False) -> List[str]:
    """Preprocess query with optional lemmatization"""

    if not query or pd.isna(query):
        return []

    text = query.lower().translate(str.maketrans("", "", string.punctuation))
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


def preprocess_queries_to_csv():
    """Preprocess queries and split into validation and test sets"""

    queries_df = load_queries("databases/voyage_3_large_suggested_queries_2048.db")

    queries_df = queries_df.dropna(subset=["query"])
    queries_df = queries_df[queries_df["query"].str.strip() != ""]
    print(f"After filtering: {len(queries_df)} valid queries")

    val_queries, test_queries = train_test_split(
        queries_df, test_size=0.85, random_state=42, shuffle=True
    )

    print(
        f"Split complete: {len(val_queries)} validation queries, {len(test_queries)} test queries"
    )

    print("Preprocessing validation queries")
    val_data = []

    with tqdm(total=len(val_queries), desc="Processing validation queries") as pbar:
        for _, row in val_queries.iterrows():
            query = row["query"]
            query_id = row["id"]

            tokens_regular = preprocess_query(query, use_lemmatization=False)

            tokens_lemmatized = preprocess_query(query, use_lemmatization=True)

            val_data.append(
                {
                    "id": query_id,
                    "query": query,
                    "tokens_regular": tokens_regular,
                    "tokens_lemmatized": tokens_lemmatized,
                }
            )

            pbar.update(1)

    print("Preprocessing test queries")
    test_data = []

    with tqdm(total=len(test_queries), desc="Processing test queries") as pbar:
        for _, row in test_queries.iterrows():
            query = row["query"]
            query_id = row["id"]

            tokens_regular = preprocess_query(query, use_lemmatization=False)
            tokens_lemmatized = preprocess_query(query, use_lemmatization=True)

            test_data.append(
                {
                    "id": query_id,
                    "query": query,
                    "tokens_regular": tokens_regular,
                    "tokens_lemmatized": tokens_lemmatized,
                }
            )

            pbar.update(1)

    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    os.makedirs("../../data", exist_ok=True)

    val_output = "../../data/validation_queries_bm25.csv"
    test_output = "../../data/test_queries_bm25.csv"

    val_df.to_csv(val_output, index=False, encoding="utf-8")
    print(f"Saved {len(val_df)} validation queries")

    test_df.to_csv(test_output, index=False, encoding="utf-8")
    print(f"Saved {len(test_df)} test queries")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total queries processed: {len(val_df) + len(test_df)}")
    print(f"Validation set: {len(val_df)} queries")
    print(f"Test set: {len(test_df)} queries")
    print(f"Validation queries saved to: {val_output}")
    print(f"Test queries saved to: {test_output}")

    return val_df, test_df


if __name__ == "__main__":
    val_df, test_df = preprocess_queries_to_csv()
