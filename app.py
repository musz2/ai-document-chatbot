import os
from dataclasses import dataclass
from typing import Iterable

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Set your OPENAI_API_KEY environment variable before asking questions.")
        st.stop()
    return OpenAI(api_key=api_key)


def extract_pdf_chunks(uploaded_files: Iterable) -> list[Chunk]:
    chunks: list[Chunk] = []

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            for chunk_text in split_text(text):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=uploaded_file.name,
                        page=page_number,
                    )
                )

    return chunks


def split_text(text: str) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_chunks(
    client: OpenAI,
    question: str,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[Chunk]:
    question_vector = embed_texts(client, [question])
    faiss.normalize_L2(question_vector)
    _, indices = index.search(question_vector, top_k)
    return [chunks[index_id] for index_id in indices[0] if index_id != -1]


def answer_question(client: OpenAI, question: str, context_chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"Source: {chunk.source}, page {chunk.page}\n{chunk.text}"
        for chunk in context_chunks
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided PDF context. "
                    "If the answer is not in the context, say you could not find it. "
                    "Cite the source filename and page when useful."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


st.set_page_config(page_title="AI Document Chatbot", layout="wide")

st.title("AI Document Chatbot")
st.caption("Upload PDFs, ask questions, and learn the basics of RAG with OpenAI embeddings and FAISS.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    build_button = st.button("Build document index", type="primary", disabled=not uploaded_files)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

if build_button:
    client = get_client()
    with st.spinner("Reading PDFs and creating embeddings..."):
        chunks = extract_pdf_chunks(uploaded_files)

        if not chunks:
            st.warning("I could not extract any text from those PDFs.")
        else:
            vectors = embed_texts(client, [chunk.text for chunk in chunks])
            st.session_state.index = build_faiss_index(vectors)
            st.session_state.chunks = chunks
            st.success(f"Indexed {len(chunks)} text chunks from {len(uploaded_files)} PDF(s).")

question = st.text_input("Ask a question about your uploaded documents")

if question:
    if st.session_state.index is None:
        st.info("Upload PDFs and build the document index first.")
    else:
        client = get_client()
        with st.spinner("Searching your documents and drafting an answer..."):
            matches = search_chunks(
                client,
                question,
                st.session_state.index,
                st.session_state.chunks,
            )
            answer = answer_question(client, question, matches)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved context"):
            for match in matches:
                st.markdown(f"**{match.source}, page {match.page}**")
                st.write(match.text)
