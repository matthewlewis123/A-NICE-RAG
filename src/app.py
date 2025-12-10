# --- Imports ---
import json
import os
import re
from datetime import datetime
from typing import Dict, List

import streamlit as st

from query_rag import RAGSystem

# --- Page Configuration and Initialization ---
st.set_page_config(page_title="NICE Clinical Assistant", layout="wide")


# Initialize RAG System
@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system"""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None


rag_system = get_rag_system()
if rag_system is None:
    st.error("RAG system failed to initialize. Please check your configuration.")
    st.stop()


# --- Helper Functions ---

# Feedback function
FEEDBACK_FILE = "feedback/query_feedback.json"
os.makedirs("feedback", exist_ok=True)


def save_feedback(query, response, sources, rating):
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "sources": sources,
        "rating": rating,
    }

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            feedback_list = json.load(f)
    else:
        feedback_list = []

    feedback_list.append(feedback)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=2)


# Display sources function
def display_sources(sources_data: List[Dict]):
    if not sources_data:
        st.markdown("No sources available for this response.")
        return

    for idx, source_info in enumerate(sources_data):
        source = source_info.get("source", "Unknown")
        section_id = source_info.get("section_id", "Unknown")
        url = source_info.get("url", "")

        clean_section = section_id
        if isinstance(section_id, str):
            if f"{source}_" in section_id:
                clean_section = section_id.replace(f"{source}_", "")

            clean_section = re.sub(r"-part\d+$", "", clean_section)
            clean_section = clean_section.replace("_", ".")

            if clean_section.startswith("Rationale and impact"):
                clean_section = "Rationale and impact"

        source_text = f"**Source {idx+1}:** {source} Section {clean_section}"
        st.markdown(source_text)

        if url:
            st.markdown(f"   🔗 [View Online Guideline]({url})")
        else:
            st.markdown("   ⚠️ No URL available")

        st.markdown("---")


# --- Session State Initialization ---
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "query_input_main" not in st.session_state:
        st.session_state.query_input_main = ""
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    if "query_to_run_next" not in st.session_state:
        st.session_state.query_to_run_next = None
    if "model_weights" not in st.session_state:
        st.session_state.model_weights = {"voyage-3-large": 1.0}
    if "similarity_k" not in st.session_state:
        st.session_state.similarity_k = 15
    if "reranker_model" not in st.session_state:
        st.session_state.reranker_model = "rerank-2"
    if "reranker_top_k" not in st.session_state:
        st.session_state.reranker_top_k = 5
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "gpt-4.1-mini"
    if "chat_data_source" not in st.session_state:
        st.session_state.chat_data_source = "NICE"


initialize_session_state()


# --- Page Styling ---
st.markdown(
    """
<style>
    .main {background-color: #f9f9f9; font-family: Arial, sans-serif;}
    h1, h2, h3, h4, h5, h6 {color: #2b6777;}
    h1 {font-weight: bold;}
    [data-testid="stSidebar"] {background-color: #e8f0fe; padding: 10px;}
    .result-box {
        border-left: 4px solid #4CAF50;
        padding: 10px;
        background-color: #fff;
        margin-bottom: 10px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div.stTextArea > div { border-radius: 8px; }
    textarea { font-family: Arial, sans-serif; font-size: 16px; color: #333; resize: vertical; }
    .stButton>button { border-radius: 5px; }
    div.stSelectbox > label {
        font-size: 16px !important;
        font-weight: bold !important;
    }
    div.stSelectbox label:first-child:contains('LLM Model') {
        font-size: 20px !important;
        font-weight: bold !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- SIDEBAR ---
with st.sidebar:
    st.header("🩺 Clinical Assistant")
    st.markdown("---")

    st.header("⚙️ Settings")

    llm_options = [
        "gpt-4.1-mini",
        "gpt-5-nano",
        "gpt-5-mini",
        "claude-sonnet-4-20250514",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    try:
        current_llm_index = llm_options.index(st.session_state.llm_model)
    except ValueError:
        current_llm_index = 0
        st.session_state.llm_model = llm_options[0]

    selected_llm = st.selectbox(
        "LLM Model", options=llm_options, key="llm_model_selector", index=current_llm_index
    )
    if selected_llm != st.session_state.llm_model:
        st.session_state.llm_model = selected_llm

    st.markdown("---")

    def new_chat_callback():
        st.session_state.chat_history = []
        st.session_state.query = ""
        st.session_state.query_input_main = ""

    if st.button("🗑️ New Chat", key="new_chat", on_click=new_chat_callback):
        pass


# --- MAIN APPLICATION AREA ---
st.title("🩺 Clinical Guidelines Chat")
st.markdown("Ask questions and get relevant information from the NICE guidelines.")


# Query processing function
def submit_and_process_query(query_to_send: str, display_query_text: str):
    st.session_state.processing_query = True

    try:
        with st.spinner("Retrieving relevant guidelines..."):
            response_chunks = []
            sources_data = []
            sources_string = ""
            temp_response_placeholder = st.empty()

            current_data_source = "NICE"

            for chunk, chunk_sources, context, chunk_sources_data in rag_system.query_rag_stream(
                query_to_send,
                st.session_state.llm_model,
                model_weights=st.session_state.model_weights,
                info_source=current_data_source,
                use_reranker=True,
                reranker_model=st.session_state.reranker_model,
                reranker_top_k=st.session_state.reranker_top_k,
                similarity_k=st.session_state.similarity_k,
            ):
                response_chunks.append(chunk)
                sources_data = chunk_sources_data
                sources_string = chunk_sources

                temp_response_placeholder.markdown(
                    f"<div style='border-left: 4px solid #4CAF50; padding-left: 10px;'>{''.join(response_chunks)}</div>",
                    unsafe_allow_html=True,
                )

            final_response = "".join(response_chunks)
            temp_response_placeholder.empty()

            st.session_state.chat_history.append(
                {
                    "query_sent": query_to_send,
                    "display_query": display_query_text,
                    "response": final_response,
                    "sources_data": sources_data,
                    "sources": sources_string,
                    "feedback_submitted": False,
                    "llm_model": st.session_state.llm_model,
                    "data_source": current_data_source,
                }
            )

    except Exception as e:
        st.error(f"Error processing query: {e}")
    finally:
        st.session_state.processing_query = False
        st.rerun()


# Display chat history
for i, chat_entry in enumerate(st.session_state.chat_history):
    st.markdown(f"👤 **You:** {chat_entry['display_query']}")

    response_info = f"(LLM: {chat_entry.get('llm_model', 'N/A')}, Source: NICE"
    if chat_entry.get("filter"):
        response_info += f", Filter: {chat_entry.get('filter')}"
    response_info += ")"

    st.markdown(f"🤖 **Assistant** {response_info}:")
    st.markdown(
        f"<div style='border-left: 4px solid #4CAF50; padding-left: 10px; margin-bottom: 10px;'>{chat_entry['response']}</div>",
        unsafe_allow_html=True,
    )

    if i == len(st.session_state.chat_history) - 1:
        if not chat_entry["feedback_submitted"]:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("👍 Helpful", key=f"helpful_{i}"):
                    save_feedback(
                        query=chat_entry["query_sent"],
                        response=chat_entry["response"],
                        sources=chat_entry.get("sources", ""),
                        rating="helpful",
                    )
                    st.session_state.chat_history[i]["feedback_submitted"] = True
                    st.toast("Thank you for your feedback!", icon="👍")
                    st.rerun()
            with action_cols[1]:
                if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                    save_feedback(
                        query=chat_entry["query_sent"],
                        response=chat_entry["response"],
                        sources=chat_entry.get("sources", ""),
                        rating="not_helpful",
                    )
                    st.session_state.chat_history[i]["feedback_submitted"] = True
                    st.toast("Thank you for your feedback!", icon="👎")
                    st.rerun()
        else:
            st.success("🙏 Feedback sent!")

        st.subheader("📚 Sources:")
        with st.expander("View Sources", expanded=False):
            sources_data = chat_entry.get("sources_data", [])
            if sources_data:
                display_sources(sources_data)
            else:
                sources_string = chat_entry.get("sources", "")
                if sources_string and sources_string.strip():
                    for idx_s, source_line in enumerate(sources_string.split("\n")):
                        source_line = source_line.strip()
                        if not source_line:
                            continue

                        st.markdown(f"**Source {idx_s+1}:** {source_line}")
                else:
                    st.markdown("No sources available for this response.")
        st.markdown("---")

# Clickable suggested queries
st.markdown("<h6>💡 Suggested Queries:</h6>", unsafe_allow_html=True)
suggested_queries_list = [
    "Which groups of people may have increased prevalence of ADHD compared with the general population?",
    "When should continuous glucose monitoring (CGM) be prescribed for patients with type 1 diabetes?",
]
sq_cols = st.columns(len(suggested_queries_list))
for idx, sq_text_item in enumerate(suggested_queries_list):
    if sq_cols[idx].button(
        sq_text_item, key=f"suggested_{idx}", disabled=st.session_state.processing_query
    ):
        st.session_state.processing_query = True
        st.session_state.query_to_run_next = sq_text_item
        st.rerun()

# Input bar
user_query = st.chat_input(
    "e.g., What are the guidelines for hypertension?",
    max_chars=1000,
    disabled=st.session_state.processing_query,
)

# Input tracker
if user_query:
    st.session_state.processing_query = True
    st.session_state.query_to_run_next = user_query
    st.rerun()

if st.session_state.get("query_to_run_next"):
    query_to_process = st.session_state.query_to_run_next
    st.session_state.query_to_run_next = None
    submit_and_process_query(query_to_process, query_to_process)
