# A NICE RAG — Retrieval-Augmented Generation over NICE Guidelines

A RAG system designed to index and retrieve National Institute for Health and Care Excellence (NICE) guideline content to generate grounded answers. 

**Note: Due to copyright restrictions, this repository does not contain any NICE guideline data. You must obtain an API key from NICE and use the provided scripts to fetch and build the dataset yourself.**

## Features

- **Interactive Streamlit App**: A chat interface for querying clinical guidelines.
- **Data Pipeline**: Scripts to download, convert, chunk, and index NICE guidelines via their official API.
- **Hybrid Retrieval**: Combines BM25 (keyword) and Vector-based (semantic) search.
- **Pluggable Architecture**:
  - **LLMs**: OpenAI (default), Gemini, Anthropic.
  - **Embeddings**: VoyageAI (optimized for retrieval).
- **Evaluation**: Integrated RAGAs framework for assessing retrieval and generation quality.

## Prerequisites

Before running the system, you need to set up the necessary API keys and environment.

### 1. API Keys
You will need API keys for the following services:
- **NICE API**: Required to download the guidelines. [Register here](https://api.nice.org.uk/).
- **OpenAI**: Required for the default generation model.
- **VoyageAI**: Required for creating and querying vector embeddings.
- **(Optional)**: Google Gemini or Anthropic keys if you wish to use those models.

### 2. Python Environment
```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
. .venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

### 3\. Environment Variables

Set your generation and embedding API keys.

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="<your openai key>"
$env:VOYAGE_API_KEY="<your voyageai key>"
# Optional
$env:GEMINI_API_KEY="<your gemini key>"
$env:ANTHROPIC_API_KEY="<your anthropic key>"

# macOS/Linux
export OPENAI_API_KEY="<your openai key>"
export VOYAGE_API_KEY="<your voyageai key>"
# Optional
export GEMINI_API_KEY="<your gemini key>"
export ANTHROPIC_API_KEY="<your anthropic key>"
```

## Data Setup (Required)

Since no data is included, you must build the database from scratch.

### Step 1: Download Guidelines

1.  Open `src/processing/nice_api_script.py`.
2.  Insert your NICE API key into the `api_key` variable:
    ```python
    api_key = "YOUR_NICE_API_KEY_HERE"
    ```
3.  Run the script to download the XML structured documents:
    ```bash
    python src/processing/nice_api_script.py
    ```
    *This will create a folder `NICE_Guidelines_XML` containing the raw files.*

### Step 2: Process and Index

Run the processing scripts in the following order to convert the data and build the search indices (Vector DB and BM25):

1.  **Convert XML to Markdown**:
    ```bash
    python src/processing/convert_xml_to_md.py
    ```
2.  **Chunk Documents**:
    ```bash
    python src/processing/chunk_mds.py
    ```
3.  **Build Vector Database** (Requires `VOYAGE_API_KEY`):
    ```bash
    python src/processing/create_database.py
    ```
4.  **Build BM25 Index**:
    ```bash
    python src/processing/preprocess_bm25.py
    ```

*Ensure the output paths in `src/config.py` match the locations where these scripts generate the databases (default is `databases/`).*

## Usage

### Interactive App

Launch the Streamlit interface:

```bash
streamlit run src/app.py
```

Use the sidebar to select your LLM model and manage settings.

### Programmatic Usage

You can run individual modules from the `src/` folder:

  - **Retrieval & Generation**: `src/query_rag.py`
  - **Retrieval Only**: `src/query_rag_retrieval.py`

## Configuration

The system behavior is controlled by `src/config.py`. You can modify this file to change:

  - **Model Weights**: Adjust the importance of Vector vs. BM25 scores.
  - **Paths**: Update locations for your local databases if you changed the ingestion pipeline.
  - **Search Parameters**: Tune `top-k` retrieval or similarity thresholds.

## Repository Structure

  - **`src/`**: Application source code.
      - **Core Modules**:
          - `app.py`: The main Streamlit web application providing the interactive chat interface.
          - `config.py`: Central configuration file for model weights, file paths, and system settings.
          - `database_manager.py`: Handles loading and caching of vector databases (SQLite) and BM25 indices.
          - `query_rag.py`: The full RAG pipeline combining retrieval and LLM generation.
          - `query_rag_retrieval.py`: A dedicated pipeline for executing and testing the retrieval stage independently.
          - `retrieval_eval.py`: Evaluation script to measure retrieval metrics (MRR, Recall) against a test set.
          - `search_engine.py`: Implements core search logic including vector similarity, BM25 scoring, rank fusion (reciprocal rank), and reranking.
      - **`processing/`**: Data ingestion and preparation scripts.
          - `nice_api_script.py`: Downloads structured guideline data (XML) directly from the NICE API.
          - `convert_xml_to_md.py`: Converts the downloaded XML files into clean, readable Markdown documents.
          - `chunk_mds.py`: Splits Markdown documents into semantic chunks suitable for embedding.
          - `create_database.py`: Generates vector embeddings (using VoyageAI) for chunks and saves them to a SQLite database.
          - `preprocess_bm25.py`: Preprocesses text (tokenization/lemmatization) to prepare for BM25 indexing.
          - `bm25_search.py`: Builds and saves the BM25 index from preprocessed data.
          - `bm25_test.py`: Runs optimization trials to find the best BM25 parameters (k1, b).
          - `preprocess_queries.py`: Prepares validation and test query sets for evaluation.
          - `preprocess_suggested.py`: Preprocesses the suggested queries used in the UI.
  - **`data/`**: Storage for raw and processed guideline text.
  - **`databases/`**: Storage for generated vector stores and indices.
  - **`ragas_modified/`**: Modified RAGAs framework for evaluation.
  - **`notebooks/`**: Jupyter notebooks for experiments and analysis.

## License and Attribution

  - **Code**: This project code is provided for educational and research purposes.
  - **Data**: The NICE guidelines are subject to the copyright and terms of use of the [National Institute for Health and Care Excellence (NICE)](https://www.nice.org.uk/). Users are responsible for adhering to NICE's API terms of service and data usage policies.
  - **Third-Party**: The `ragas_modified/` folder contains code derived from the [Ragas project](https://github.com/explodinggradients/ragas).
