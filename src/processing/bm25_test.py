# --- Imports
import ast
import json
import os
import pickle
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from skopt import gp_minimize
from skopt.space import Categorical, Real
from skopt.utils import use_named_args
from tqdm import tqdm

STOPWORDS_SET = set(stopwords.words("english"))

for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)


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


def load_preprocessed_queries(csv_path: str) -> pd.DataFrame:
    """Load preprocessed queries from CSV"""
    df = pd.read_csv(csv_path, encoding="utf-8")

    def safe_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return []

    df["tokens_regular"] = df["tokens_regular"].apply(safe_eval)
    df["tokens_lemmatized"] = df["tokens_lemmatized"].apply(safe_eval)

    return df


def create_bm25_index(
    chunks_df: pd.DataFrame,
    k1: float,
    b: float,
    epsilon: float,
    tokenization_type: str,
    cache_dir: str,
) -> Tuple[BM25Okapi, List[str]]:
    """Create BM25 index with caching"""

    config_str = f"k1-{k1:.3f}_b-{b:.3f}_eps-{epsilon:.3f}_tok-{tokenization_type}"
    cache_file = os.path.join(cache_dir, f"bm25_{config_str}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                tqdm.write(f"  ✓ Loaded cached index: {config_str}")
                return data["bm25"], data["section_ids"]
        except:
            tqdm.write(f"  ⚠ Failed to load cache, rebuilding...")

    tqdm.write(
        f"Creating BM25 index with k1={k1:.3f}, b={b:.3f}, epsilon={epsilon:.3f}, tokenization={tokenization_type}..."
    )

    token_column = "tokens_lemmatized" if tokenization_type == "lemmatized" else "tokens_regular"
    corpus = chunks_df[token_column].tolist()
    section_ids = chunks_df["id"].tolist()

    valid_indices = [i for i, tokens in enumerate(corpus) if tokens and len(tokens) > 0]
    corpus = [corpus[i] for i in valid_indices]
    section_ids = [section_ids[i] for i in valid_indices]

    bm25 = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tqdm.write(f"  Saving BM25 index to cache...")
        with open(cache_file, "wb") as f:
            pickle.dump({"bm25": bm25, "section_ids": section_ids}, f)
        tqdm.write(f"  ✓ Index cached successfully")
    except Exception as e:
        tqdm.write(f"  ⚠ Failed to cache index: {e}")

    return bm25, section_ids


def search_bm25(query_tokens: List[str], bm25: BM25Okapi, section_ids: List[str]) -> List[str]:
    """Perform BM25 search using preprocessed query tokens"""
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:25]
    return [section_ids[i] for i in ranked_indices]


def evaluate_bm25(
    chunks_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    k1: float,
    b: float,
    epsilon: float,
    tokenization_type: str,
    cache_dir: str,
) -> Dict[str, float]:
    """Evaluate BM25 configuration using preprocessed data"""

    tqdm.write(
        f"Creating BM25 index (k1={k1:.3f}, b={b:.3f}, eps={epsilon:.3f}, tok={tokenization_type})..."
    )
    bm25, section_ids = create_bm25_index(chunks_df, k1, b, epsilon, tokenization_type, cache_dir)

    tqdm.write(f"Starting evaluation on {len(queries_df)} queries...")
    ranks = []

    token_column = "tokens_lemmatized" if tokenization_type == "lemmatized" else "tokens_regular"

    with tqdm(
        total=len(queries_df),
        desc=f"Evaluating queries",
        ncols=120,
        unit="q",
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    ) as pbar:

        for idx, row in queries_df.iterrows():
            query_tokens = row[token_column]
            results = search_bm25(query_tokens, bm25, section_ids)
            try:
                rank = results.index(row["id"]) + 1
                ranks.append(rank)
            except ValueError:
                ranks.append(len(section_ids) + 1)

            if len(ranks) % 50 == 0:
                current_recall_15 = sum(1 for r in ranks if r <= 15) / len(ranks)
                pbar.set_postfix(
                    {"R@15": f"{current_recall_15:.3f}", "Avg_Rank": f"{np.mean(ranks):.1f}"}
                )

            pbar.update(1)

    tqdm.write(f"Evaluation complete. Calculating metrics")

    total = len(ranks)
    metrics = {
        "recall_at_1": sum(1 for r in ranks if r <= 1) / total,
        "recall_at_5": sum(1 for r in ranks if r <= 5) / total,
        "recall_at_10": sum(1 for r in ranks if r <= 10) / total,
        "recall_at_15": sum(1 for r in ranks if r <= 15) / total,
        "mrr": np.mean([1.0 / r for r in ranks]),
        "mean_rank": np.mean([r for r in ranks]),
    }

    print(f"Metrics calculated")
    return metrics


def main():
    print("Loading preprocessed chunks")
    chunks_df = load_preprocessed_chunks("data/chunks_bm25.csv")
    print(f"Loaded {len(chunks_df)} preprocessed chunks")

    print("Loading preprocessed queries")
    val_queries = load_preprocessed_queries("data/validation_queries_bm25.csv")
    test_queries = load_preprocessed_queries("data/test_queries_bm25.csv")
    print(f"Loaded {len(val_queries)} validation queries, {len(test_queries)} test queries")

    print("=" * 50)

    search_space = [
        Real(0.5, 3.0, name="k1"),
        Real(0.1, 1.0, name="b"),
        Real(0.05, 0.5, name="epsilon"),
        Categorical(["regular", "lemmatized"], name="tokenization_type"),
    ]

    results = []

    pbar = tqdm(
        total=50,
        desc="Bayesian Optimization",
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    @use_named_args(search_space)
    def objective(k1, b, epsilon, tokenization_type):
        tqdm.write(f"\n--- Trial {len(results) + 1}/50 ---")
        tqdm.write(
            f"Testing parameters: k1={k1:.4f}, b={b:.4f}, epsilon={epsilon:.4f}, tokenization={tokenization_type}"
        )

        metrics = evaluate_bm25(
            chunks_df, val_queries, k1, b, epsilon, tokenization_type, "../../results/bm25_cache"
        )

        result = {
            "k1": k1,
            "b": b,
            "epsilon": epsilon,
            "tokenization_type": tokenization_type,
            **metrics,
        }
        results.append(result)

        current_best = max(results, key=lambda x: x["recall_at_15"])["recall_at_15"]
        pbar.set_postfix(
            {
                "Current_R@15": f"{metrics['recall_at_15']:.4f}",
                "Best_R@15": f"{current_best:.4f}",
                "Trial": f"{len(results)}",
            }
        )
        pbar.update(1)

        tqdm.write(f"Trial {len(results)} complete")
        tqdm.write(
            f"Result: Recall@15={metrics['recall_at_15']:.4f}, MRR={metrics['mrr']:.4f}, Mean_Rank={metrics['mean_rank']:.1f}"
        )
        tqdm.write("-" * 40)

        return -metrics["recall_at_15"]

    print("Starting Bayesian optimization")
    gp_minimize(objective, search_space, n_calls=50, n_initial_points=10, random_state=42)
    pbar.close()

    print("\n" + "=" * 50)
    print("Optimization complete")
    print("=" * 50)

    results_df = pd.DataFrame(results).sort_values("recall_at_15", ascending=False)
    best = results_df.iloc[0]

    print(f"Best config from validation:")
    print(f"  k1={best['k1']:.4f}, b={best['b']:.4f}, eps={best['epsilon']:.4f}")
    print(f"  tokenization={best['tokenization_type']}")
    print(f"  Validation Recall@15: {best['recall_at_15']:.4f}")

    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION")
    print("=" * 50)
    print("Evaluating ONLY the best config on test set...")
    test_metrics = evaluate_bm25(
        chunks_df,
        test_queries,
        best["k1"],
        best["b"],
        best["epsilon"],
        best["tokenization_type"],
        "../../results/bm25_cache",
    )

    print(f"\nFinal Test Results:")
    print(f"Test Recall@1: {test_metrics['recall_at_1']:.4f}")
    print(f"Test Recall@5: {test_metrics['recall_at_5']:.4f}")
    print(f"Test Recall@10: {test_metrics['recall_at_10']:.4f}")
    print(f"Test Recall@15: {test_metrics['recall_at_15']:.4f}")
    print(f"Test MRR: {test_metrics['mrr']:.4f}")
    print(f"Test Mean Rank: {test_metrics['mean_rank']:.2f}")

    os.makedirs("../../results", exist_ok=True)
    results_df.to_csv("../../results/bm25_optimization_results.csv", index=False)

    with open("../../results/bm25_test_results.json", "w") as f:
        json.dump(
            {
                "best_config": {
                    "k1": float(best["k1"]),
                    "b": float(best["b"]),
                    "epsilon": float(best["epsilon"]),
                    "tokenization_type": best["tokenization_type"],
                },
                "validation_metrics": dict(
                    best[
                        [
                            "recall_at_1",
                            "recall_at_5",
                            "recall_at_10",
                            "recall_at_15",
                            "mrr",
                            "mean_rank",
                        ]
                    ]
                ),
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print("Saved test results to JSON")

    return results_df, test_metrics


if __name__ == "__main__":
    results_df, test_metrics = main()
