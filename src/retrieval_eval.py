# --- Imports
import ast
import os
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from query_rag_retrieval import RetrievalEvaluationSystem

# --- Data Loading and Preparation ---


def load_queries_from_db(db_path: str) -> pd.DataFrame:
    """Load queries and embeddings from a SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM queries WHERE query_embedding IS NOT NULL;", conn)
    conn.close()
    df["query_embedding"] = df["query_embedding"].apply(
        lambda x: np.frombuffer(x, dtype=np.float32)
    )
    return df.reset_index(drop=True)


def load_bm25_preprocessed_queries(csv_path: str) -> pd.DataFrame:
    """Load preprocessed queries for BM25 optimization"""
    df = pd.read_csv(csv_path, encoding="utf-8")

    def safe_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return []

    df["tokens_lemmatized"] = df["tokens_lemmatized"].apply(safe_eval)

    return df


# --- Evaluation Core ---


class RetrievalEvaluator:
    def __init__(self, db_paths: Dict[str, str]):
        self.db_paths = db_paths
        self.retrieval_system = RetrievalEvaluationSystem()

    def evaluate_query(
        self,
        query: str,
        expected_id: str,
        query_embeddings: Dict[str, np.ndarray],
        params: Dict,
        query_tokens: Optional[List[str]] = None,
    ) -> Dict:
        try:
            results = self.retrieval_system.retrieve_documents(
                query_text=query,
                query_embeddings=query_embeddings,
                query_tokens=query_tokens,
                similarity_k=params["similarity_k"],
                common_sections_n=params["common_sections_n"],
                info_source=params.get("info_source", "NICE"),
                model_weights=params["model_weights"],
                filename_type_filter=params.get("filename_type_filter"),
                use_hybrid_search=params["use_hybrid_search"],
                use_reranker=params.get("use_reranker", False),
                reranker_model=params.get("reranker_model", "rerank-2"),
                reranker_top_k=params.get("reranker_top_k"),
                wrrf_k=params["wrrf_k"],
            )
            docs = results[0] if isinstance(results, tuple) else results
            rank = -1
            for i, doc in enumerate(docs):
                doc_id = doc.get("section_id") if isinstance(doc, dict) else doc
                if doc_id == expected_id:
                    rank = i + 1
                    break
            return {"rank": rank, "found": rank > 0, "total_retrieved": len(docs)}
        except Exception as e:
            return {"rank": -1, "found": False, "total_retrieved": 0, "error": str(e)}


# --- Metrics ---


def calculate_metrics(results: List[Dict]) -> Dict:
    found = [r for r in results if r.get("found")]
    found_ranks = [r["rank"] for r in found]

    all_ranks = []
    for r in results:
        if r.get("found"):
            all_ranks.append(r["rank"])
        else:
            all_ranks.append(100000)

    mrr = sum(1.0 / r["rank"] for r in found) / len(results) if results else 0.0
    recall_at = lambda k: sum(1 for r in found if r["rank"] <= k) / len(results) if results else 0.0

    return {
        "total": len(results),
        "found": len(found),
        "success_rate": len(found) / len(results) if results else 0.0,
        "mean_rank": np.mean(found_ranks) if found_ranks else None,
        "median_rank": np.median(found_ranks) if found_ranks else None,
        "max_rank": np.max(all_ranks) if all_ranks else None,
        "mrr": mrr,
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@15": recall_at(15),
    }


# --- Main Evaluation ---


def main():
    db_paths = {
        "voyage-3-large": "databases/voyage_3_large_suggested_queries_2048.db",
        "voyage-3.5": "databases/voyage_3.5_suggested_queries_2048.db",
        "text-embedding-3-large": "databases/text_embedding_3_large_suggested_queries.db",
        "Qwen3": "databases/Qwen3-Embedding-0.6B_suggested_queries.db",
    }

    # Define all configurations to test
    configurations = [
        {
            "name": "Voyage-3-Large",
            "model_weights": {
                "voyage-3-large": 1.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 0.0,
                "BM25": 0.0,
            },
            "use_hybrid_search": False,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "Voyage-3.5",
            "model_weights": {
                "voyage-3-large": 0.0,
                "voyage-3.5": 1.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 0.0,
                "BM25": 0.0,
            },
            "use_hybrid_search": False,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "Text-Embedding-3-Large",
            "model_weights": {
                "voyage-3-large": 0.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 1.0,
                "Qwen3": 0.0,
                "BM25": 0.0,
            },
            "use_hybrid_search": False,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "Qwen3-Embedding-0.6B",
            "model_weights": {
                "voyage-3-large": 0.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 1.0,
                "BM25": 0.0,
            },
            "use_hybrid_search": False,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "BM25",
            "model_weights": {
                "voyage-3-large": 0.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 0.0,
                "BM25": 1.0,
            },
            "use_hybrid_search": True,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "Voyage-3-Large + BM25",
            "model_weights": {
                "voyage-3-large": 5.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 0.0,
                "BM25": 1.0,
            },
            "use_hybrid_search": True,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "Voyage-3-Large + Text-Embedding-3-Large",
            "model_weights": {
                "voyage-3-large": 2.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 1.0,
                "Qwen3": 0.0,
                "BM25": 0.0,
            },
            "use_hybrid_search": False,
            "similarity_k": 12000,
            "common_sections_n": 12000,
        },
        {
            "name": "Voyage-3-Large + BM25 (Reranker 2 Lite)",
            "model_weights": {
                "voyage-3-large": 5.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 0.0,
                "BM25": 1.0,
            },
            "use_hybrid_search": True,
            "use_reranker": True,
            "reranker_model": "rerank-2-lite",
            "reranker_top_k": 10,
            "similarity_k": 25,
            "common_sections_n": 15,
        },
        {
            "name": "Voyage-3-Large + BM25 (Reranker 2)",
            "model_weights": {
                "voyage-3-large": 5.0,
                "voyage-3.5": 0.0,
                "text-embedding-3-large": 0.0,
                "Qwen3": 0.0,
                "BM25": 1.0,
            },
            "use_hybrid_search": True,
            "use_reranker": True,
            "reranker_model": "rerank-2",
            "reranker_top_k": 10,
            "similarity_k": 25,
            "common_sections_n": 15,
        },
    ]

    evaluator = RetrievalEvaluator(db_paths)

    # Load all databases
    cached_dbs = {}
    print("Loading databases...")
    for model, db_path in db_paths.items():
        cached_dbs[model] = load_queries_from_db(db_path)
        print(f"  Loaded {model}: {len(cached_dbs[model])} queries")

    # Load preprocessed queries for BM25
    full_preprocessed = load_bm25_preprocessed_queries(
        "data/suggested_queries_bm25_preprocessed.csv"
    )
    print(f"  Loaded BM25 preprocessed: {len(full_preprocessed)} queries")

    # Create train/validation split (using the same random seed for reproducibility)
    train_indices, val_indices = train_test_split(
        range(len(cached_dbs["voyage-3-large"])), test_size=0.15, random_state=42, shuffle=True
    )

    # Base parameters (same for all configurations except similarity_k and common_sections_n)
    base_params = {
        "wrrf_k": 40,
        "filename_type_filter": "CG,NG",
    }

    # Prepare output file
    output_file = "results/retrieval_evaluation_results.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write CSV header only if file doesn't exist
    file_exists = os.path.exists(output_file)
    if not file_exists:
        with open(output_file, "w") as f:
            f.write(
                "Model,MRR,Recall@1,Recall@5,Recall@10,Recall@15,Median_Rank,Mean_Rank,Max_Rank\n"
            )

    # Evaluate each configuration
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config['name']}")
        print(f"{'='*60}")

        model_weights = config["model_weights"]
        use_hybrid_search = config["use_hybrid_search"]
        similarity_k = config["similarity_k"]
        common_sections_n = config["common_sections_n"]

        # Determine active embedding models
        active_embedding_models = [
            k for k in model_weights if k != "BM25" and model_weights.get(k, 0) > 0
        ]

        if not active_embedding_models:
            # BM25 only - use voyage-3-large as reference for query list
            main_model = "voyage-3-large"
            queries_df = cached_dbs[main_model].copy()
            embeddings = {main_model: queries_df["query_embedding"].values}
        else:
            # Select main model (with highest weight)
            main_model = max(active_embedding_models, key=lambda k: model_weights[k])
            queries_df = cached_dbs[main_model].copy()
            embeddings = {main_model: queries_df["query_embedding"].values}

            # Merge other active models
            for model in active_embedding_models:
                if model == main_model or model_weights[model] <= 0:
                    continue
                other_df = cached_dbs[model]
                merged = queries_df.merge(
                    other_df[["id", "query_embedding"]],
                    on="id",
                    how="left",
                    suffixes=("", f"_{model}"),
                )
                embeddings[model] = merged[f"query_embedding_{model}"].values
                queries_df = merged[merged[f"query_embedding_{model}"].notna()].copy()

        # Prepare test set
        test_queries_df = queries_df.iloc[train_indices].reset_index(drop=True)
        test_embeddings = {k: v[train_indices] for k, v in embeddings.items()}

        # Fix: Align preprocessed tokens by ID, not by positional index
        test_preprocessed_df = full_preprocessed[
            full_preprocessed["id"].isin(test_queries_df["id"])
        ]
        test_preprocessed_lookup = {
            row["id"]: row["tokens_lemmatized"] for _, row in test_preprocessed_df.iterrows()
        }

        need_bm25_tokens = model_weights.get("BM25", 0) > 0 and use_hybrid_search

        # Set up parameters for this configuration
        params = base_params.copy()
        params["model_weights"] = model_weights
        params["use_hybrid_search"] = use_hybrid_search
        params["similarity_k"] = similarity_k
        params["common_sections_n"] = common_sections_n

        print(f"Model weights: {model_weights}")
        print(f"Use hybrid search: {use_hybrid_search}")
        print(f"Similarity K: {similarity_k}")
        print(f"Common sections N: {common_sections_n}")
        print(f"Test set size: {len(test_queries_df)}")

        # Evaluate
        test_results = []
        for idx, row in tqdm(
            enumerate(test_queries_df.itertuples(index=False)),
            total=len(test_queries_df),
            desc=f"Evaluating {config['name']}",
        ):
            query_embeddings = {
                k: v[idx] for k, v in test_embeddings.items() if idx < len(v) and v[idx] is not None
            }
            query_tokens = test_preprocessed_lookup.get(row.id) if need_bm25_tokens else None
            eval_result = evaluator.evaluate_query(
                row.query, row.id, query_embeddings, params, query_tokens
            )
            test_results.append(eval_result)

        # Calculate metrics
        test_metrics = calculate_metrics(test_results)

        # Print results (handle None values)
        print(f"\nResults for {config['name']}:")
        print(f"  MRR: {test_metrics['mrr']:.3f}")
        print(f"  Recall@1: {test_metrics['recall@1']:.3f}")
        print(f"  Recall@5: {test_metrics['recall@5']:.3f}")
        print(f"  Recall@10: {test_metrics['recall@10']:.3f}")
        print(f"  Recall@15: {test_metrics['recall@15']:.3f}")

        median_rank_display = (
            f"{test_metrics['median_rank']}" if test_metrics["median_rank"] is not None else "N/A"
        )
        mean_rank_display = (
            f"{test_metrics['mean_rank']:.3f}" if test_metrics["mean_rank"] is not None else "N/A"
        )

        print(f"  Median Rank: {median_rank_display}")
        print(f"  Mean Rank: {mean_rank_display}")
        print(f"  Max Rank: {test_metrics['max_rank']}")
        print(f"  Success Rate: {test_metrics['success_rate']:.4f}")

        # Write this result to CSV immediately (handle None values)
        mean_rank_str = (
            f"{test_metrics['mean_rank']:.3f}" if test_metrics["mean_rank"] is not None else "N/A"
        )
        median_rank_str = (
            str(test_metrics["median_rank"]) if test_metrics["median_rank"] is not None else "N/A"
        )

        with open(output_file, "a") as f:
            f.write(
                f"{config['name']},{test_metrics['mrr']:.3f},{test_metrics['recall@1']:.3f},"
                f"{test_metrics['recall@5']:.3f},{test_metrics['recall@10']:.3f},"
                f"{test_metrics['recall@15']:.3f},{median_rank_str},"
                f"{mean_rank_str},{test_metrics['max_rank']}\n"
            )

        print(f"Result written to {output_file}")

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_file}")
    print(f"{'='*60}")

    # Read and display final summary
    results_df = pd.read_csv(output_file)
    print("\nSummary Table:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
