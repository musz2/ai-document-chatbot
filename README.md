# AI Document Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) project.

Upload PDFs, turn their text into OpenAI embeddings, store those embeddings in a simple FAISS vector index, and ask questions about the uploaded documents in a Streamlit app.

## What You Will Learn

- LLM basics
- Embeddings
- Vector search
- RAG fundamentals
- PDF ingestion
- Streamlit app structure

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Run the app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## How It Works

1. You upload one or more PDF files.
2. The app extracts text from each page.
3. Text is split into overlapping chunks.
4. Each chunk is converted into an embedding with OpenAI.
5. FAISS stores the embeddings for fast similarity search.
6. Your question is embedded and matched against the most relevant chunks.
7. The OpenAI chat model answers using the retrieved PDF context.

## Notes

- This is an in-memory beginner project. The FAISS index resets when the app restarts.
- Text extraction works best with PDFs that contain selectable text.
- Scanned image-only PDFs need OCR, which is intentionally outside this beginner version.
