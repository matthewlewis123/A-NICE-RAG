# --- Imports ---
import logging
import os
import pickle
import sqlite3
import threading
from typing import Tuple

import numpy as np
import pandas as pd


# --- Database Manager Class ---
class DatabaseManager:

    def __init__(self):
        self._embeddings_cache = {}
        self._bm25_cache = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def load_embeddings_from_sql(self, db_path: str, model_name: str = None) -> pd.DataFrame:
        """Load embeddings from SQL database."""

        cache_key = f"{db_path}_{model_name}" if model_name else db_path

        with self._lock:
            if cache_key in self._embeddings_cache:
                return self._embeddings_cache[cache_key]

        try:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found: {db_path}")

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT id, content, source, embedding, url FROM chunks")
            rows = cursor.fetchall()

            if not rows:
                self.logger.warning(f"No chunks found in {db_path}")
                return pd.DataFrame()

            records = []
            for row in rows:
                try:
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                    records.append(
                        {
                            "id": row["id"],
                            "document": row["content"],
                            "source": row["source"],
                            "embedding": embedding,
                            "url": row["url"],
                        }
                    )
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Skipping invalid row {row['id']}: {e}")
                    continue

            df = pd.DataFrame(records)

            with self._lock:
                self._embeddings_cache[cache_key] = df

            return df

        except Exception as e:
            self.logger.error(f"Error loading embeddings from {db_path}: {e}")
            raise
        finally:
            if "conn" in locals():
                conn.close()

    def load_bm25_from_pickle(self, filepath: str) -> Tuple:
        """Load BM25 index from pickle file."""
        with self._lock:
            if filepath in self._bm25_cache:
                return self._bm25_cache[filepath]

        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"BM25 index not found: {filepath}")

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            result = (data["bm25"], data["sections"], data["section_ids"])

            with self._lock:
                self._bm25_cache[filepath] = result

            return result

        except Exception as e:
            self.logger.error(f"Error loading BM25 index from {filepath}: {e}")
            raise
