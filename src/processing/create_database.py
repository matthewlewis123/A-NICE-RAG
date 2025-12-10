# --- Imports ---
import json
import os
import sqlite3
from typing import Any, Dict, List

import numpy as np
import voyageai
from openai import OpenAI
from tqdm import tqdm


def get_voyage_client():
    """Get Voyage AI client with API key from environment."""

    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    return voyageai.Client(api_key=voyage_api_key)


def get_openai_client():
    """Get OpenAI client with API key from environment."""

    openai_api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=openai_api_key)


def get_embeddings_batch(texts: List[str], client, model: str = "voyage-3.5") -> List[List[float]]:
    """Get embeddings for a batch of texts using Voyage or OpenAI API."""

    try:
        if "voyage" in model:
            result = client.embed(
                texts=texts,
                model=model,
                input_type="document",
                output_dimension=2048,
                truncation=True,
            )
            return result.embeddings

        else:
            response = client.embeddings.create(input=texts, model=model)
            embeddings = [item.embedding for item in response.data]
            return embeddings

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise


def init_database(db_path: str):
    """Initialize the SQLite database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        source TEXT NOT NULL,
        embedding BLOB
    )
    """
    )

    conn.commit()
    conn.close()


def load_chunked_data(file_path: str) -> List[Dict]:
    """Load chunked data from JSON file."""

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if "chunks" in data:
        return data["chunks"]

    return [data]


def get_existing_chunk_ids(db_path: str) -> set:
    """Get IDs of chunks that already exist in the database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM chunks")
    existing_ids = {row[0] for row in cursor.fetchall()}
    conn.close()

    return existing_ids


def store_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    content: str,
    source: str,
    embedding: List[float] = None,
):
    """Store a chunk in the database."""

    cursor = conn.cursor()

    embedding_blob = None
    if embedding:
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

    cursor.execute(
        """
    INSERT OR REPLACE INTO chunks (id, content, source, embedding)
    VALUES (?, ?, ?, ?)
    """,
        (chunk_id, content, source, embedding_blob),
    )

    conn.commit()


def create_embeddings_db(
    chunks_file: str, model_info: Dict[str, Any], db_name: str = "nice_guidelines"
):
    """Create or update a SQLite database with document embeddings."""

    model_name = model_info["model_name"]
    voyage_client = model_info["client"]

    db_path = f"../../databases/{model_name.replace('-', '_')}_{db_name}_2048.db"

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    init_database(db_path)

    chunks = load_chunked_data(chunks_file)
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    conn = sqlite3.connect(db_path)

    print(f"\nProcessing chunks for model: {model_name}")
    print(f"Database: {db_path}")

    existing_ids = get_existing_chunk_ids(db_path)
    chunks_to_process = []

    for chunk in chunks:
        chunk_id = chunk.get("title", "")

        if not chunk_id or chunk_id in existing_ids:
            continue

        chunks_to_process.append(
            {"id": chunk_id, "content": chunk.get("content", ""), "source": chunk.get("source", "")}
        )

    if not chunks_to_process:
        print(f"No new chunks to process")
        conn.close()
        return

    print(f"Processing embeddings for {len(chunks_to_process)} chunks")

    batch_size = 100
    for i in tqdm(range(0, len(chunks_to_process), batch_size), desc="Processing batches"):
        batch = chunks_to_process[i : i + batch_size]
        texts = [chunk["content"] for chunk in batch]

        try:
            embeddings = get_embeddings_batch(texts, voyage_client, model_name)

            for chunk_data, embedding in zip(batch, embeddings):
                store_chunk(
                    conn, chunk_data["id"], chunk_data["content"], chunk_data["source"], embedding
                )

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            continue

    conn.close()
    print(f"Database creation/update completed: {db_path}")


def main():
    """Create Database"""

    voyage_client = get_voyage_client()
    # openai_client = get_openai_client()

    embedding_models = [
        {"model_name": "voyage-3.5", "client": voyage_client},
    ]

    chunks_file = "../../data/chunked_guidelines.json"

    if not os.path.exists(chunks_file):
        print(f"Error: File not found: {chunks_file}")
        return

    print(f"Processing chunks from: {chunks_file}")

    for model_info in embedding_models:
        try:
            create_embeddings_db(chunks_file, model_info)
            print(f"Completed processing for {model_info['model_name']}")
        except Exception as e:
            print(f"Error processing {model_info['model_name']}: {e}")
            continue


if __name__ == "__main__":
    main()
