# from google import genai
# from google.genai import types

import ollama

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import bm25s

# load_dotenv()
# client = genai.Client()
# EMBED_MODEL = "gemini-embedding-001"
# EMBED_DIM = 3072

EMBED_MODEL = "qwen3-embedding:4b"
EMBED_DIM = 2560

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

# -------------------------------------------------
# 2. Dense embeddings
# -------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:

    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM,
    )

    return response.embeddings

def embed_query(query: str) -> list[float]:

    task = (
        "Given a user question, retrieve relevant passages "
        "from the uploaded documents that answer the question"
    )

    instructed_query = (
        f"Instruct: {task}\n"
        f"Query: {query}"
    )

    response = ollama.embed(
        model=EMBED_MODEL,
        input=instructed_query,
        dimensions=EMBED_DIM,
    )

    return response.embeddings[0]

# -------------------------------------------------
# 3. BM25 index
# -------------------------------------------------
def build_bm25_index(chunks: list[str]):
    tokens = bm25s.tokenize(
        chunks,
        stopwords="en"
    )

    retriever = bm25s.BM25()
    retriever.index(tokens)

    return retriever

# -------------------------------------------------
# 4. BM25 search
# -------------------------------------------------
def search_bm25(query: str, retriever, chunks: list[str], k: int = 5):

    query_tokens = bm25s.tokenize(query, stopwords="en")
    indices, scores = retriever.retrieve(query_tokens,k=k)

    results = []
    for rank, index in enumerate(indices[0]):
        results.append({
            "text": chunks[index],
            "score": float(scores[0][rank])
        })
    return results