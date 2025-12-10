# --- Imports ---
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import voyageai

from config import Config, InfoSource
from database_manager import DatabaseManager
from search_engine import SearchEngine

# Disable logging
logging.getLogger().disabled = True
logger = logging.getLogger(__name__)
logger.disabled = True


# --- Retrieval System Class ---
class RetrievalEvaluationSystem:

    def __init__(self):
        """Initialize the retrieval system components"""

        self.config = Config()
        self.db_manager = DatabaseManager()

        try:
            self.voyage_client = voyageai.Client()
            self.search_engine = SearchEngine(self.voyage_client, None)
        except Exception as e:
            logger.warning(f"Failed to initialize Voyage client: {e}. Reranking will be disabled.")
            self.voyage_client = None
            self.search_engine = SearchEngine(None, None)

        self._load_databases()

    def _load_databases(self):
        """Load embedding databases."""
        start_time = time.time()
        logger.info("Loading databases...")

        self.embeddings_data = {}
        self.bm25_data = {}

        for source in InfoSource:
            source_config = self.config.SOURCE_CONFIGS[source]
            try:
                embeddings_dict = {}

                # Load Voyage-3-Large embeddings
                if hasattr(source_config, "voyage_db_path") and source_config.voyage_db_path:
                    voyage_df = self.db_manager.load_embeddings_from_sql(
                        source_config.voyage_db_path, "voyage-3-large"
                    )
                    embeddings_dict["voyage-3-large"] = voyage_df
                    logger.info(
                        f"Loaded {source.value} Voyage-3-Large embeddings: {len(voyage_df)} documents"
                    )

                # Load Voyage 3.5 embeddings
                if (
                    hasattr(source_config, "voyage_3_5_db_path")
                    and source_config.voyage_3_5_db_path
                ):
                    voyage_3_5_df = self.db_manager.load_embeddings_from_sql(
                        source_config.voyage_3_5_db_path, "voyage-3.5"
                    )
                    embeddings_dict["voyage-3.5"] = voyage_3_5_df
                    logger.info(
                        f"Loaded {source.value} Voyage 3.5 embeddings: {len(voyage_3_5_df)} documents"
                    )

                # Load OpenAI embeddings
                if hasattr(source_config, "openai_db_path") and source_config.openai_db_path:
                    openai_df = self.db_manager.load_embeddings_from_sql(
                        source_config.openai_db_path, "text-embedding-3-large"
                    )
                    embeddings_dict["text-embedding-3-large"] = openai_df
                    logger.info(
                        f"Loaded {source.value} OpenAI embeddings: {len(openai_df)} documents"
                    )

                # Load Qwen embeddings
                if hasattr(source_config, "qwen_db_path") and source_config.qwen_db_path:
                    qwen_df = self.db_manager.load_embeddings_from_sql(
                        source_config.qwen_db_path, "Qwen3-Embedding-0.6B"
                    )
                    embeddings_dict["Qwen3"] = qwen_df
                    logger.info(f"Loaded {source.value} Qwen embeddings: {len(qwen_df)} documents")

                self.embeddings_data[source] = embeddings_dict

                # Load BM25 data
                bm25_tuple = self.db_manager.load_bm25_from_pickle(source_config.bm25_path)
                self.bm25_data[source] = bm25_tuple

                total_embeddings = sum(
                    len(df) for df in embeddings_dict.values() if df is not None and not df.empty
                )
                logger.info(
                    f"Loaded {source.value}: {total_embeddings} total embeddings across {len(embeddings_dict)} models"
                )

            except Exception as e:
                logger.error(f"Failed to load {source.value}: {e}")
                self.embeddings_data[source] = {}
                self.bm25_data[source] = None

        end_time = time.time()
        logger.info(f"Database loading completed in {end_time - start_time:.2f} seconds")

    def _validate_inputs(
        self,
        query_embeddings: Dict[str, np.ndarray],
        similarity_k: int,
        common_sections_n: int,
        info_source: str,
    ):
        """Validate input parameters"""
        if not query_embeddings:
            raise ValueError("Query embeddings dictionary cannot be empty")

        for model_name, embedding in query_embeddings.items():
            if not isinstance(embedding, np.ndarray):
                raise ValueError(f"Embedding for {model_name} must be a numpy array")
            if embedding.size == 0:
                raise ValueError(f"Embedding for {model_name} cannot be empty")

        if similarity_k <= 0 or common_sections_n <= 0:
            raise ValueError("similarity_k and common_sections_n must be positive integers")

        try:
            InfoSource(info_source.lower())
        except ValueError:
            valid_sources = [s.value for s in InfoSource]
            raise ValueError(
                f"Invalid info_source '{info_source}'. Must be one of: {valid_sources}"
            )

    def get_sources(self, results: List[Tuple], info_source: str) -> List[str]:
        """Extract only section_id from results"""
        section_ids = []
        for doc, score in results:
            section_id = doc.get("id", "Unknown section")
            section_ids.append(section_id)
        return section_ids

    def retrieve_documents(
        self,
        query_embeddings: Dict[str, np.ndarray],
        query_text: Optional[str] = None,
        query_tokens: Optional[List[str]] = None,
        similarity_k: int = 25,
        common_sections_n: int = 15,
        info_source: str = "NICE",
        model_weights: Optional[Dict[str, float]] = None,
        filename_type_filter: Optional[str] = None,
        use_hybrid_search: bool = False,
        wrrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model: str = "rerank-2-lite",
        reranker_top_k: Optional[int] = 5,
        return_docs: bool = False,
    ) -> Tuple[List[Dict], Dict]:

        self._validate_inputs(query_embeddings, similarity_k, common_sections_n, info_source)

        if model_weights is None:
            model_weights = self.config.DEFAULT_MODEL_WEIGHTS.copy()

        source_enum = InfoSource(info_source.lower())

        logger.info(f"Processing retrieval for source: {info_source}")
        if filename_type_filter:
            logger.info(f"Filtering by guideline type: {filename_type_filter}")

        try:
            embeddings_dict = self.embeddings_data.get(source_enum, {})
            bm25_tuple = self.bm25_data.get(source_enum)

            if not embeddings_dict:
                logger.warning(f"No embedding data available for source: {info_source}")
                return []

            if bm25_tuple is None:
                logger.warning(f"No BM25 data available for source: {info_source}")
                bm25, bm25_sections, bm25_section_ids = bm25_tuple if bm25_tuple else (None, [], [])
            else:
                bm25, bm25_sections, bm25_section_ids = bm25_tuple

            bm25_section_dict = {section.metadata["id"]: section for section in bm25_sections}

            ranked_lists = []
            all_results = []

            # Search with Voyage embeddings
            voyage_df = embeddings_dict.get("voyage-3-large")
            if (
                voyage_df is not None
                and not voyage_df.empty
                and model_weights.get("voyage-3-large", 0) > 0
                and "voyage-3-large" in query_embeddings
            ):

                voyage_results = self.search_engine.similarity_search_with_embedding(
                    query_embeddings["voyage-3-large"],
                    voyage_df,
                    "voyage-3-large",
                    similarity_k,
                    filename_type_filter,
                )
                if not voyage_results.empty:
                    voyage_ranked_list = voyage_results["id"].tolist()
                    ranked_lists.append((voyage_ranked_list, "voyage-3-large"))

                    voyage_docs = voyage_results.to_dict("records")
                    all_results.extend(list(zip(voyage_results["id"], voyage_docs)))

                    logger.info(f"Voyage search found {len(voyage_results)} results")

            # Search with Voyage 3.5 embeddings
            voyage_3_5_df = embeddings_dict.get("voyage-3.5")
            if (
                voyage_3_5_df is not None
                and not voyage_3_5_df.empty
                and model_weights.get("voyage-3.5", 0) > 0
                and "voyage-3.5" in query_embeddings
            ):

                voyage_3_5_results = self.search_engine.similarity_search_with_embedding(
                    query_embeddings["voyage-3.5"],
                    voyage_3_5_df,
                    "voyage-3.5",
                    similarity_k,
                    filename_type_filter,
                )
                if not voyage_3_5_results.empty:
                    voyage_3_5_ranked_list = voyage_3_5_results["id"].tolist()
                    ranked_lists.append((voyage_3_5_ranked_list, "voyage-3.5"))

                    existing_ids = {doc_id for doc_id, _ in all_results}
                    filtered_voyage_3_5_results = voyage_3_5_results[
                        ~voyage_3_5_results["id"].isin(existing_ids)
                    ]
                    voyage_3_5_docs = filtered_voyage_3_5_results.to_dict("records")
                    all_results.extend(
                        list(zip(filtered_voyage_3_5_results["id"], voyage_3_5_docs))
                    )

                    logger.info(f"Voyage 3.5 search found {len(voyage_3_5_results)} results")

            # Search with OpenAI embeddings
            openai_df = embeddings_dict.get("text-embedding-3-large")
            if (
                openai_df is not None
                and not openai_df.empty
                and model_weights.get("text-embedding-3-large", 0) > 0
                and "text-embedding-3-large" in query_embeddings
            ):

                openai_results = self.search_engine.similarity_search_with_embedding(
                    query_embeddings["text-embedding-3-large"],
                    openai_df,
                    "text-embedding-3-large",
                    similarity_k,
                    filename_type_filter,
                )
                if not openai_results.empty:
                    openai_ranked_list = openai_results["id"].tolist()
                    ranked_lists.append((openai_ranked_list, "text-embedding-3-large"))

                    existing_ids = {doc_id for doc_id, _ in all_results}
                    filtered_openai_results = openai_results[
                        ~openai_results["id"].isin(existing_ids)
                    ]
                    openai_docs = filtered_openai_results.to_dict("records")
                    all_results.extend(list(zip(filtered_openai_results["id"], openai_docs)))
                    logger.info(f"OpenAI search found {len(openai_results)} results")

            # Search with Qwen embeddings
            qwen_df = embeddings_dict.get("Qwen3")
            if (
                qwen_df is not None
                and not qwen_df.empty
                and model_weights.get("Qwen3", 0) > 0
                and "Qwen3" in query_embeddings
            ):

                qwen_results = self.search_engine.similarity_search_with_embedding(
                    query_embeddings["Qwen3"], qwen_df, "Qwen3", similarity_k, filename_type_filter
                )
                if not qwen_results.empty:
                    qwen_ranked_list = qwen_results["id"].tolist()
                    ranked_lists.append((qwen_ranked_list, "Qwen3"))

                    existing_ids = {doc_id for doc_id, _ in all_results}
                    filtered_qwen_results = qwen_results[~qwen_results["id"].isin(existing_ids)]
                    qwen_docs = filtered_qwen_results.to_dict("records")
                    all_results.extend(list(zip(filtered_qwen_results["id"], qwen_docs)))
                    logger.info(f"Qwen search found {len(qwen_results)} results")

            # Search with BM25
            if use_hybrid_search and bm25 is not None and model_weights.get("BM25", 0) > 0:
                bm25_ranked_list = None

                if query_tokens:
                    bm25_ranked_list = self.search_engine.bm25_search_preprocessed(
                        query_tokens,
                        bm25,
                        bm25_sections,
                        bm25_section_ids,
                        similarity_k,
                        filename_type_filter,
                    )
                    logger.info(f"BM25 search using preprocessed tokens")
                elif query_text:
                    bm25_ranked_list = self.search_engine.bm25_search(
                        query_text,
                        bm25,
                        bm25_sections,
                        bm25_section_ids,
                        similarity_k,
                        filename_type_filter,
                    )
                    logger.info(f"BM25 search using query text")
                else:
                    logger.warning(
                        "BM25 search requested but no query_text or query_tokens provided - skipping BM25"
                    )

                if bm25_ranked_list:
                    ranked_lists.append((bm25_ranked_list, "BM25"))
                    logger.info(f"BM25 search found {len(bm25_ranked_list)} results")

                    existing_ids = {doc_id for doc_id, _ in all_results}

                    all_results.extend(
                        (
                            section_id,
                            {
                                "id": section_id,
                                "document": section.page_content,
                                "source": section.metadata.get("source", "Unknown"),
                                "similarity": 0.0,
                            },
                        )
                        for section_id in bm25_ranked_list
                        if section_id not in existing_ids
                        and (section := bm25_section_dict.get(section_id))
                    )
            elif use_hybrid_search and model_weights.get("BM25", 0) > 0 and not query_text:
                logger.warning("BM25 search requested but no query_text provided - skipping BM25")

            # Perform weighted reciprocal rank fusion
            if len(ranked_lists) > 1:
                fused_ranks = self.search_engine.weighted_reciprocal_rank_fusion(
                    ranked_lists, model_weights, wrrf_k
                )
                most_common_sections = [
                    section_id for section_id, _ in fused_ranks[:common_sections_n]
                ]
                logger.info(f"Fused results from {len(ranked_lists)} ranking methods using WRRF")
            elif len(ranked_lists) == 1:
                ranked_list, model_name = ranked_lists[0]
                most_common_sections = ranked_list[:common_sections_n]
                logger.info(f"Using single ranking method: {model_name}")
            else:
                most_common_sections = []
                logger.warning("No ranking methods available - no sections selected")

            if all_results:
                results_dict = {doc_id: doc for doc_id, doc in all_results}
                common_docs = [
                    results_dict[section_id]
                    for section_id in most_common_sections
                    if section_id in results_dict
                ][:common_sections_n]
            else:
                common_docs = []

            # Apply reranking
            if use_reranker and common_docs and len(common_docs) > 1 and query_text:
                logger.info(f"Applying reranking to {len(common_docs)} fused results")
                common_docs = self.search_engine.rerank_documents(
                    query_text, common_docs, reranker_model, reranker_top_k
                )
                logger.info(f"Post-fusion reranking completed with model '{reranker_model}'")
            elif use_reranker and not query_text:
                logger.info("Reranking requested but no query_text provided - skipping reranking")
            elif use_reranker:
                logger.info("Reranking skipped: insufficient results for reranking")
            else:
                logger.info("Reranking disabled")

            final_results = [
                (doc, doc.get("rerank_score", doc.get("similarity", 0.0))) for doc in common_docs
            ]

            if return_docs:
                return common_docs

            sources_data = self.get_sources(final_results, info_source)

            logger.info(f"Retrieved {len(final_results)} documents")

            return sources_data

        except Exception as e:
            logger.error(f"Error in retrieval processing: {e}")
            return []
