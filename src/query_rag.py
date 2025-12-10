# --- Imports ---
import logging
import os
import re
import time
import traceback
from typing import Dict, Generator, List, Optional, Tuple

import voyageai
from openai import OpenAI

from config import Config, InfoSource
from database_manager import DatabaseManager
from search_engine import SearchEngine

# Enable logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- RAG System Class ---
class RAGSystem:

    def __init__(self):
        """Initialize the RAG system components"""

        self.config = Config()
        self.db_manager = DatabaseManager()
        self.embeddings_data = {}
        self.bm25_data = {}

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            self.gemini_client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            self.gemini_client = None

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.anthropic_client = OpenAI(
                api_key=anthropic_api_key, base_url="https://api.anthropic.com/v1/"
            )
        else:
            self.anthropic_client = None

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            self.openrouter_client = OpenAI(
                api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1/"
            )
        else:
            self.openrouter_client = None

        self.voyage_client = voyageai.Client()
        self.search_engine = SearchEngine(self.voyage_client, self.openai_client)

        self._load_databases()

    def _load_databases(self):
        """Load embedding databases."""

        start_time = time.time()
        logger.info("Loading databases")

        self.embeddings_data = {}

        for source in InfoSource:
            source_config = self.config.SOURCE_CONFIGS[source]
            try:
                embeddings_dict = {}

                # Load Voyage embeddings
                if hasattr(source_config, "voyage_db_path") and source_config.voyage_db_path:
                    voyage_df = self.db_manager.load_embeddings_from_sql(
                        source_config.voyage_db_path, "voyage-3-large"
                    )
                    embeddings_dict["voyage-3-large"] = voyage_df
                    logger.info(
                        f"Loaded {source.value} Voyage embeddings: {len(voyage_df)} documents"
                    )

                self.embeddings_data[source] = embeddings_dict

                # Load BM25 data
                try:
                    bm25_tuple = self.db_manager.load_bm25_from_pickle(source_config.bm25_path)
                    self.bm25_data[source] = bm25_tuple
                    if bm25_tuple:
                        bm25, bm25_sections, bm25_section_ids = bm25_tuple
                        logger.info(
                            f"Loaded {source.value} BM25 data: {len(bm25_sections)} sections"
                        )
                    else:
                        logger.warning(f"No BM25 data loaded for {source.value}")
                except Exception as bm25_error:
                    logger.error(f"Failed to load BM25 data for {source.value}: {bm25_error}")
                    self.bm25_data[source] = None

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
        self, query_text: str, similarity_k: int, common_sections_n: int, info_source: str
    ):
        """Validate the inputs for the RAG system query."""

        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        if similarity_k <= 0:
            raise ValueError("similarity_k must be a positive integer")

        if common_sections_n <= 0:
            raise ValueError("common_sections_n must be a positive integer")

        try:
            InfoSource(info_source.lower())
        except ValueError:
            valid_sources = [s.value for s in InfoSource]
            raise ValueError(
                f"Invalid info_source '{info_source}'. Must be one of: {valid_sources}"
            )

    def _get_context_text(self, reranked_results: List[Tuple]) -> Tuple[str, List[str]]:
        """Extract context text and section names from retrieved results."""

        context_text_sections = []
        raw_sections = []

        for doc, _ in reranked_results:
            section_id = doc.get("id", "Unknown section")
            source = doc.get("source", "Unknown file")

            clean_section_id = self._clean_section_id(section_id, source)

            document_text = doc["document"]
            raw_sections.append(document_text)

            formatted_section = (
                f"Source Information: [Identifier: {source} - Section ID: {clean_section_id}]\n"
                f"Context: {document_text}"
            )
            context_text_sections.append(formatted_section)

        return "\n\n---\n\n".join(context_text_sections), "\n\n--- Context Chunk ---\n\n".join(
            raw_sections
        )

    def _create_system_prompt(self, context_text: str, query_text: str) -> List[Dict]:
        """Create the system prompt for the RAG model."""

        return [
            {
                "role": "system",
                "content": (
                    "You are a medical AI assistant tasked with answering clinical questions strictly based on the provided NICE clinical guidelines context. Follow the requirements below to ensure accurate, consistent, and professional responses.\n\n"
                    "# Response Rules\n\n"
                    "1. **Context Restriction**:\n"
                    "   - Only use information given in the provided NICE guidelines context.\n"
                    "   - Do not generate or speculate with information not explicitly found in the given context.\n\n"
                    "2. **Answer Format**:\n"
                    "   - Provide a clear and concise response based solely on the context.\n"
                    "   - When including a list, use standard markdown bullet points (`*` or `-`).\n"
                    "   - If a list follows introductory text, insert a line break before the first bullet point.\n"
                    "   - Each bullet point must be on its own line.\n\n"
                    "3. **Preserve Tables**:\n"
                    "   - If relevant markdown tables appear in the context, reproduce them in your answer.\n"
                    "   - Maintain the original structure, formatting, and content of any included tables.\n\n"
                    "4. **Links and URLs**:\n"
                    "   - Include any URLs or web links from the context directly in your response when relevant.\n"
                    "   - Integrate links naturally within sentences, using markdown syntax for clickable text links.\n"
                    "   - DO NOT generate or invent any URLs not explicitly present in the context.\n\n"
                    "5. **Markdown Link Formatting**:\n"
                    "   - In responses, only the descriptive text in brackets should be visible and clickable (e.g., `[NICE hypertension guidelines](https://www.nice.org.uk/guidance/ng136)`).\n"
                    "   - Readers should never see raw URLs in the text.\n\n"
                    "6. **If No Relevant Information**:\n"
                    "   - If the context contains no relevant information, state clearly:\n"
                    '      *"No relevant NICE guidelines were found."*\n\n'
                    "# Output Format\n\n"
                    "- All responses should be in plain text, using markdown formatting for lists and links as required.\n"
                    "- Do not use code blocks.\n"
                    "- Answers should be concise, accurate, and formatted according to the rules above.\n\n"
                    "# Examples\n\n"
                    "**Example 1: Integration of markdown link in context**\n"
                    'Question: "What is the recommended treatment for stage 2 hypertension?"\n'
                    "Context snippet: ...see the [NICE hypertension guidelines](https://www.nice.org.uk/guidance/ng136)...\n"
                    "Output:\n"
                    "According to the [NICE hypertension guidelines](https://www.nice.org.uk/guidance/ng136), stage 2 hypertension should be treated with...\n\n"
                    "**Example 2: Multiple guideline references**\n"
                    "According to these guidelines:\n"
                    "* Initial treatment is lifestyle modification.\n"
                    "* For persistent hypertension, refer to [hypertension treatment guideline](https://www.nice.org.uk/guidance/ng97).\n\n"
                    "**Example 3: No relevant context**\n"
                    "No relevant NICE guidelines were found.\n\n"
                    "# Notes\n\n"
                    "- Never output information beyond what is provided in the supplied context.\n"
                    "- Always use markdown for lists and links.\n"
                    "- Make sure all markdown tables from context are preserved in your answer if relevant.\n"
                    "- Present links only as clickable text, not as bare URLs.\n\n"
                    "**REMINDER:**\n"
                    "Strictly adhere to all formatting and content rules above for every response."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{query_text}\n\n"
                    f"Context from NICE clinical guidelines:\n"
                    f"{context_text}"
                ),
            },
        ]

    def get_sources_from_retrieved(self, reranked_results: List[Tuple]) -> str:
        """Extract source information from retrieved results."""

        sources = []
        for doc, _ in reranked_results:
            section_id = doc.get("id", "Unknown section")
            source = doc.get("source", "Unknown file")
            clean_section_id = self._clean_section_id(section_id, source)
            sources.append(f"{source} - Section {clean_section_id}")

        return "\n".join(sources)

    def _clean_section_id(self, section_id: str, source: str) -> str:
        """Clean the section ID for display purposes."""

        if not section_id or section_id == "Unknown section":
            return section_id

        clean_section = section_id

        if clean_section.startswith(f"{source}_"):
            clean_section = clean_section[len(f"{source}_") :]

        clean_section = clean_section.replace("_", " ")
        clean_section = re.sub(r"\s+", " ", clean_section).strip()

        return clean_section

    def query_rag_stream(
        self,
        query_text: str,
        llm_model: str,
        similarity_k: int = 25,
        common_sections_n: int = 15,
        wrrf_k: int = 40,
        info_source: str = "NICE",
        model_weights: Optional[Dict[str, float]] = None,
        filename_type_filter: Optional[str] = None,
        use_hybrid_search: bool = False,
        use_reranker: bool = True,
        reranker_model: str = "rerank-2",
        reranker_top_k: Optional[int] = 10,
    ) -> Generator[Tuple[str, str, str, List[Dict]], None, None]:
        """Query the RAG system and return a stream of results."""

        self._validate_inputs(query_text, similarity_k, common_sections_n, info_source)

        if model_weights is None:
            model_weights = self.config.DEFAULT_MODEL_WEIGHTS.copy()

        start_time = time.time()
        source_enum = InfoSource(info_source.lower())

        logger.info(f"Processing query for source: {info_source}")

        # Retrieve Results
        try:
            embeddings_dict = self.embeddings_data.get(source_enum, {})
            bm25_tuple = self.bm25_data.get(source_enum)

            if not embeddings_dict:
                logger.warning(f"No embedding data available for source: {info_source}")
                yield f"Error: No data available for source {info_source}", "", ""
                return

            if bm25_tuple is None:
                logger.warning(f"No BM25 data available for source: {info_source}")
                bm25, bm25_sections, bm25_section_ids = bm25_tuple if bm25_tuple else (None, [], [])
            else:
                bm25, bm25_sections, bm25_section_ids = bm25_tuple

            bm25_section_dict = {section.metadata["id"]: section for section in bm25_sections}

            ranked_lists = []
            all_results = []

            voyage_df = embeddings_dict.get("voyage-3-large")
            if (
                voyage_df is not None
                and not voyage_df.empty
                and model_weights.get("voyage-3-large", 0) > 0
            ):

                voyage_results = self.search_engine.similarity_search(
                    query_text, voyage_df, "voyage-3-large", similarity_k, filename_type_filter
                )
                if not voyage_results.empty:
                    voyage_ranked_list = voyage_results["id"].tolist()
                    ranked_lists.append((voyage_ranked_list, "voyage-3-large"))
                    voyage_docs = voyage_results.to_dict("records")
                    all_results.extend(list(zip(voyage_results["id"], voyage_docs)))

                    logger.info(f"Voyage search found {len(voyage_results)} results")

            # Search with BM25
            if use_hybrid_search and bm25 is not None and model_weights.get("BM25", 0) > 0:
                bm25_ranked_list = None

                if query_text:
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

            if use_reranker and common_docs and len(common_docs) > 1:
                logger.info(f"Applying reranking to {len(common_docs)} fused results")
                common_docs = self.search_engine.rerank_documents(
                    query_text, common_docs, reranker_model, reranker_top_k
                )
                logger.info(f"Post-fusion reranking completed with model '{reranker_model}'")
            elif use_reranker:
                logger.info("Reranking skipped: insufficient results for reranking")
            else:
                logger.info("Reranking disabled")

            reranked_results = []
            for doc in common_docs:
                score = doc.get("rerank_score", doc.get("similarity", 0.0))
                reranked_results.append((doc, score))

            retrieval_time = time.time() - start_time
            logger.info(f"Retrieval completed in {retrieval_time:.4f} seconds")
            logger.info(f"Final context contains {len(reranked_results)} sections for LLM")

            context_text, raw_sections = self._get_context_text(reranked_results)
            system_prompt = self._create_system_prompt(context_text, query_text)

            sources_string = self.get_sources_from_retrieved(reranked_results)

            sources_data = []
            for doc, score in reranked_results:
                source_info = {
                    "source": doc.get("source", "Unknown"),
                    "section_id": doc.get("id", "Unknown"),
                    "url": doc.get("url", "Unknown"),
                }
                sources_data.append(source_info)

            llm_start_time = time.time()

            # Different LLM Model Handling
            try:
                if "gemini" in llm_model.lower() and self.gemini_client:
                    stream = self.gemini_client.chat.completions.create(
                        model=llm_model, messages=system_prompt, temperature=0, stream=True
                    )

                    for chunk in stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            yield content, sources_string, raw_sections, sources_data

                elif "gpt-4" in llm_model.lower() and self.openai_client:
                    stream = self.openai_client.chat.completions.create(
                        model=llm_model, messages=system_prompt, temperature=0, stream=True
                    )

                    for chunk in stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            yield content, sources_string, raw_sections, sources_data

                elif "gpt-5" in llm_model.lower() and self.openai_client:
                    stream = self.openai_client.chat.completions.create(
                        model=llm_model,
                        messages=system_prompt,
                        reasoning_effort="minimal",
                        stream=True,
                    )

                    for chunk in stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            yield content, sources_string, raw_sections, sources_data

                elif "o4-mini" in llm_model.lower() and self.openai_client:
                    stream = self.openai_client.chat.completions.create(
                        model=llm_model, messages=system_prompt, stream=True
                    )

                    for chunk in stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            yield content, sources_string, raw_sections, sources_data

                elif "claude" in llm_model.lower() and self.openrouter_client:
                    stream = self.openrouter_client.chat.completions.create(
                        model=llm_model, messages=system_prompt, temperature=0, stream=True
                    )

                    for chunk in stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            yield content, sources_string, raw_sections, sources_data

                else:
                    error_msg = f"Unsupported LLM model or client not available: {llm_model}"
                    logger.error(error_msg)
                    yield error_msg, "", ""
                    return

            except Exception as e:
                logger.error(f"Error in LLM completion: {e}")
                yield f"Error generating response: {str(e)}", "", ""
                return

            finally:
                llm_time = time.time() - llm_start_time
                print()
                logger.info(f"LLM completion time: {llm_time:.4f} seconds")

        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            yield f"Error processing query: {str(e)}", "", ""


def main():
    """Main function for CLI usage"""
    # query_text = "What groups are at higher risk of ADHD?"
    query_text = "What are the recommended interventions for smoking cessation in adults?"
    llm_model = "gemini-2.5-flash"
    similarity_k = 15
    common_sections_n = 15
    info_source = "NICE"
    filename_type_filter = "CG,NG"
    use_hybrid_search = False
    use_reranker = True
    reranker_top_k = 10

    try:
        print("Initializing RAG system...")
        rag_system = RAGSystem()

        print(f"\n=== Query: {query_text} ===")
        print(f"Source: {info_source}")
        print(f"LLM Model: {llm_model}")
        print(f"Using hybrid search: {use_hybrid_search}")
        print(f"Using reranker: {use_reranker}")
        print("\n=== LLM Response ===\n")

        response_text = ""
        formatted_sources = ""

        for chunk, sources, context, sources_data in rag_system.query_rag_stream(
            query_text=query_text,
            llm_model=llm_model,
            similarity_k=similarity_k,
            common_sections_n=common_sections_n,
            info_source=info_source,
            filename_type_filter=filename_type_filter,
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker,
            reranker_top_k=reranker_top_k,
        ):
            print(chunk, end="", flush=True)
            response_text += chunk
            formatted_sources = sources
        print(f"\n\n=== Sources ===")
        print(formatted_sources)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
