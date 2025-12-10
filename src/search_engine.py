# --- Imports ---
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import voyageai

from processing.preprocess_bm25 import preprocess_text


# --- Search Engine Class ---
class SearchEngine:

    def __init__(self, voyage_client: voyageai.Client, openai_client=None):
        self.vo = voyage_client
        self.openai_client = openai_client
        self.logger = logging.getLogger(__name__)

    def weighted_reciprocal_rank_fusion(
        self, ranked_lists: List[Tuple], model_weights: Dict[str, float], k: int = 50
    ) -> List[Tuple]:
        """Perform weighted reciprocal rank fusion on a multiple ranked lists."""
        rrf_scores = defaultdict(float)

        for ranked_list, model_name in ranked_lists:
            weight = model_weights.get(model_name, 1.0)
            _ = [
                rrf_scores.__setitem__(doc_id, rrf_scores[doc_id] + weight * (1 / (k + rank)))
                for rank, doc_id in enumerate(ranked_list, start=1)
            ]

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _filter_by_filename_type(self, df: pd.DataFrame, filename_type_filter: str) -> pd.DataFrame:
        """Filter DataFrame by filename type."""

        prefixes = tuple(p.strip().upper() for p in filename_type_filter.split(","))
        source_upper = df["source"].str.upper()

        if len(prefixes) == 1:
            mask = source_upper.str.startswith(prefixes[0], na=False)
        else:
            pattern = "^(?:" + "|".join(prefixes) + ")"
            mask = source_upper.str.contains(pattern, na=False, regex=True)

        filtered_df = df[mask].copy()

        prefix_str = ", ".join(prefixes)
        self.logger.info(
            f"Filtered by filename type(s) '{prefix_str}': {len(filtered_df)} documents remaining from {len(df)} total"
        )

        return filtered_df

    def similarity_search_with_embedding(
        self,
        query_embedding: np.ndarray,
        df: pd.DataFrame,
        model_name: str = "voyage-3-large",
        similarity_k: int = 25,
        filename_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform similarity search with pre-calculated dense embedding."""

        try:
            if filename_type_filter:
                df = self._filter_by_filename_type(df, filename_type_filter)

            if df.empty:
                self.logger.warning(
                    f"No documents found after filtering by filename type: {filename_type_filter}"
                )
                return df

            query_embedding = (
                query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
            )
            embeddings = np.stack(df["embedding"].values)
            similarities = np.dot(query_embedding, embeddings.T).flatten()

            if len(similarities) > similarity_k:
                top_k_indices = np.argpartition(similarities, -similarity_k)[-similarity_k:]
                top_k_indices = top_k_indices[similarities[top_k_indices].argsort()[::-1]]
            else:
                top_k_indices = similarities.argsort()[::-1]

            result_df = df.iloc[top_k_indices].copy()
            result_df["similarity"] = similarities[top_k_indices]

            return result_df

        except Exception as e:
            self.logger.error(
                f"Error in {model_name} similarity search with precalculated embedding: {e}"
            )
            return pd.DataFrame()

    def similarity_search(
        self,
        query_text: str,
        df: pd.DataFrame,
        model_name: str = "voyage-3-large",
        similarity_k: int = 25,
        filename_type_filter: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Perform similarity search on the DataFrame, calculating embeddings for query."""

        try:
            if filename_type_filter:
                df = self._filter_by_filename_type(df, filename_type_filter)

            if df.empty:
                self.logger.warning(
                    f"No documents found after filtering by filename type: {filename_type_filter}"
                )
                return df

            if query_embedding is not None:
                query_embedding = query_embedding.reshape(1, -1)
                self.logger.info(f"Using provided pre-calculated {model_name} query embedding")
            else:
                query_embedding = self._generate_query_embedding(query_text, model_name)
                self.logger.info(f"Generated new {model_name} query embedding")

            embeddings = np.stack(df["embedding"].values)
            similarities = np.dot(query_embedding, embeddings.T).flatten()

            if len(similarities) > similarity_k:
                top_indices = np.argpartition(similarities, -similarity_k)[-similarity_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1]

            result_df = df.iloc[top_indices].copy()
            result_df["similarity"] = similarities[top_indices]

            self.logger.info(f"{model_name} similarity search found {len(result_df)} results")

            return result_df

        except Exception as e:
            self.logger.error(f"Error in {model_name} similarity search: {e}")
            return pd.DataFrame()

    def _generate_query_embedding(self, query_text: str, model_name: str) -> np.ndarray:
        """Generate query embedding using the specified model."""

        if model_name == "voyage-3-large":
            if not self.vo:
                raise ValueError("Voyage client not available")
            response = self.vo.embed(
                query_text, input_type="query", model="voyage-3-large", output_dimension=2048
            ).embeddings
            return np.array(response).reshape(1, -1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def rerank_documents(
        self,
        query_text: str,
        documents: List,
        reranker_model: str = "rerank-2",
        reranker_top_k: Optional[int] = None,
    ) -> List:
        """Rerank documents based on their relevance to the query using specified reranker."""

        try:
            if not documents:
                return documents

            document_texts = [doc.get("document", "") for doc in documents]

            self.logger.info(
                f"Starting reranking with model '{reranker_model}' for {len(document_texts)} documents"
            )
            reranking_result = self.vo.rerank(
                query=query_text,
                documents=document_texts,
                model=reranker_model,
                top_k=reranker_top_k or len(document_texts),
                truncation=True,
            )

            reranked_documents = [
                {**documents[idx], "rerank_score": score}
                for idx, score in zip(
                    (result.index for result in reranking_result.results),
                    (result.relevance_score for result in reranking_result.results),
                )
                if idx < len(documents)
            ]

            self.logger.info(
                f"Reranking completed: {len(reranked_documents)} documents reordered by relevance"
            )
            return reranked_documents

        except Exception as e:
            self.logger.warning(f"Reranking failed, returning original order: {e}")
            return documents

    def _core_bm25_search(
        self,
        query_tokens: List[str],
        bm25,
        bm25_sections,
        bm25_section_ids,
        similarity_k: int,
        filename_type_filter: Optional[str],
    ) -> List[str]:
        """Perform BM25 search on the document sections."""

        if not query_tokens:
            return []

        bm25_scores = bm25.get_scores(query_tokens)

        if filename_type_filter:
            prefixes = tuple(p.strip().upper() for p in filename_type_filter.split(","))

            filtered_indices_scores = [
                (i, bm25_scores[i])
                for i, section in enumerate(bm25_sections)
                if any(
                    section.metadata.get("source", "").upper().startswith(prefix)
                    for prefix in prefixes
                )
            ]

            top_k = sorted(filtered_indices_scores, key=lambda x: x[1], reverse=True)[:similarity_k]
            return [bm25_section_ids[idx] for idx, _ in top_k]
        else:
            scores_array = np.array(bm25_scores)
            if len(scores_array) > similarity_k:
                top_indices = np.argpartition(scores_array, -similarity_k)[-similarity_k:]
                top_indices = top_indices[scores_array[top_indices].argsort()[::-1]]
            else:
                top_indices = scores_array.argsort()[::-1]

            return [bm25_section_ids[i] for i in top_indices]

    def bm25_search(
        self,
        query_text: str,
        bm25,
        bm25_sections,
        bm25_section_ids,
        similarity_k: int = 25,
        filename_type_filter: Optional[str] = None,
        use_lemmatized: bool = True,
    ) -> List[str]:
        """Perform BM25 search with preprocessing of query."""

        try:
            query_tokens = preprocess_text(query_text, use_lemmatization=use_lemmatized)
            return self._core_bm25_search(
                query_tokens,
                bm25,
                bm25_sections,
                bm25_section_ids,
                similarity_k,
                filename_type_filter,
            )
        except Exception as e:
            self.logger.error(f"Error in BM25 search: {e}")
            return []

    def bm25_search_preprocessed(
        self,
        query_tokens: List[str],
        bm25,
        bm25_sections,
        bm25_section_ids,
        similarity_k: int = 25,
        filename_type_filter: Optional[str] = None,
    ) -> List[str]:
        """Perform BM25 search with no preprocessing of query."""

        try:
            return self._core_bm25_search(
                query_tokens,
                bm25,
                bm25_sections,
                bm25_section_ids,
                similarity_k,
                filename_type_filter,
            )
        except Exception as e:
            self.logger.error(f"Error in preprocessed BM25 search: {e}")
            return []
