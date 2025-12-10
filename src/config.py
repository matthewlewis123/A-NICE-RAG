# --- Imports ---
from dataclasses import dataclass
from enum import Enum


# Information sources
class InfoSource(Enum):
    NICE = "nice"


# Individual Source Configuration
@dataclass
class SourceConfig:
    db_path: str
    bm25_path: str
    context_description: str
    not_found_message: str
    voyage_db_path: str = None
    voyage_3_5_db_path: str = None
    openai_db_path: str = None
    qwen_db_path: str = None

    def __post_init__(self):
        if self.voyage_db_path is None:
            self.voyage_db_path = self.db_path


# Global Configuration
class Config:
    DEFAULT_MODEL_WEIGHTS = {
        "voyage-3-large": 5.0,
        "text-embedding-3-large": 0.0,
        "voyage-3.5": 0.0,
        "Qwen3": 0.0,
        "BM25": 1.0,
    }

    SOURCE_CONFIGS = {
        InfoSource.NICE: SourceConfig(
            db_path="databases/voyage_3_large_nice_guidelines_2048.db",
            bm25_path="databases/bm25_index_nice_guidelines.pkl",
            context_description="NICE guidelines",
            not_found_message="no relevant NICE guidelines were found",
            voyage_db_path="databases/voyage_3_large_nice_guidelines_2048.db",
            voyage_3_5_db_path="databases/voyage_3.5_nice_guidelines_2048.db",
            openai_db_path="databases/text_embedding_3_large_nice_guidelines.db",
            qwen_db_path="databases/Qwen3-Embedding-0.6B_nice_guidelines.db",
        )
    }

    @classmethod
    def get_source_config(cls, source: str) -> SourceConfig:
        try:
            source_enum = InfoSource(source.lower())
            return cls.SOURCE_CONFIGS[source_enum]
        except ValueError:
            raise ValueError(
                f"Unknown source: {source}. Valid sources: {[s.value for s in InfoSource]}"
            )
