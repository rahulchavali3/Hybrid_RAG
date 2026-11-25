import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental.ai import gemini
from dotenv import load_dotenv
import uuid
import os
import datetime
from pathlib import Path
import json

from data_loader import load_and_chunk_pdf, embed_texts, embed_query
from vector_db import QdrantStorage
from custom_types import (RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc,)

load_dotenv()

# --------------------------------------------------------------
# Inngest client
# --------------------------------------------------------------
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

# ==============================================================
# INGEST PDF
# ==============================================================

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2,
        period=datetime.timedelta(minutes=1),
    ),
    # rate_limit=inngest.RateLimit(
    #     limit=1,
    #     period=datetime.timedelta(hours=4),
    #     key="event.data.source_id",
    # ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    # ----------------------------------------------------------
    # Step 1: Load PDF and chunk it
    # ---------------------------------------------------------
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:

        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id") or Path(pdf_path).name

        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id,
        )
    # ----------------------------------------------------------
    # Step 2: Create dense embeddings + store BM25 text
    # ----------------------------------------------------------
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Dense embeddings
        dense_vecs = embed_texts(chunks)

        # Stable ID for every chunk
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{i}",
                )
            )
            for i in range(len(chunks))
        ]

        # Original chunk text + source
        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        store = QdrantStorage()
        # Store BOTH:
        # 1. dense embedding
        # 2. BM25/sparse representation of the chunk text
        store.upsert_hybrid(ids=ids, dense_vectors=dense_vecs, texts=chunks, payloads=payloads,)

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc,)

    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult,)

    return ingested.model_dump()


# ==============================================================
# QUERY PDF
# ==============================================================

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai"
    ),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    question = ctx.event.data["question"]
    source_ids = ctx.event.data["source_ids"]

    top_k = int(ctx.event.data.get("top_k", 5))

    # Gemini setup
    adapter = gemini.Adapter(
        auth_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-3.6-flash",
    )
    # ----------------------------------------------------------
    # Step 1: Generate 3 rewritten queries
    # ----------------------------------------------------------
    rewrite_prompt = (
        "You are a search-query rewriting assistant for a RAG system.\n"
        "Generate exactly 3 different search queries for document retrieval.\n"
        "Keep the original meaning.\n"
        "Do not answer the question.\n\n"
        f"Question: {question}"
    )

    rewrite_res = await ctx.step.ai.infer(
        "rewrite-queries",
        adapter=adapter,
        body={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": rewrite_prompt}
                    ],
                }
            ],
            "generationConfig": {

                # Give Gemini enough room
                "maxOutputTokens": 1024,

                # Query rewriting is simple.
                # We don't need heavy reasoning.
                "thinkingConfig": {
                    "thinkingLevel": "minimal"
                },

                # Force JSON output
                "responseMimeType": "application/json",

                "responseSchema": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
        },
    )

    rewrite_text = (
        rewrite_res["candidates"][0]
        ["content"]
        ["parts"][0]
        ["text"]
        .strip()
    )

    # print("[RAW REWRITE]", repr(rewrite_text))

    rewritten_queries = json.loads(rewrite_text)

    # print("[MULTI-QUERY]", rewritten_queries)

    # ----------------------------------------------------------
    # Step 2: Search using all 3 queries + RRF
    # ----------------------------------------------------------

    def _multi_search(queries: list[str], source_ids: list[str],top_k: int = 5,) -> RAGSearchResult:

        store = QdrantStorage()
        all_results = []

        # Run hybrid search for every rewritten query
        for query in queries:

            query_vec = embed_query(query)
            found = store.hybrid_search(
                query_vector=query_vec,
                query_text=query,
                source_ids=source_ids,
                top_k=top_k,
            )

            results = list(
                zip(
                    found["contexts"],
                    found["sources"],
                )
            )

            all_results.append(results)

        # ------------------------------------------------------
        # RRF across the 3 query result lists
        # ------------------------------------------------------

        rrf_scores = {}
        rrf_k = 60

        for results in all_results:

            for rank, (context, source) in enumerate(
                results,
                start=1,
            ):
                # Same chunk = same key
                key = (source, context)

                rrf_scores[key] = (rrf_scores.get(key, 0) + 1 / (rrf_k + rank))

        # Sort chunks by final RRF score
        ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True,)[:top_k]

        return RAGSearchResult(
            contexts=[context for source, context in ranked],
            sources=[source for source, context in ranked],
        )

    found = await ctx.step.run(
        "multi-query-hybrid-search",
        lambda: _multi_search(
            rewritten_queries,
            source_ids,
            top_k,
        ),
        output_type=RAGSearchResult,
    )

    # ----------------------------------------------------------
    # Step 3: Give retrieved chunks to Gemini
    # ----------------------------------------------------------

    context_block = "\n\n".join(
        f"- {c}"
        for c in found.contexts
    )

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # ----------------------------------------------------------
    # Step 4: Generate final answer
    # ----------------------------------------------------------

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "system_instruction": {
                "parts": [
                    {
                        "text":
                        "You answer questions using only "
                        "the provided context."
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_content}
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 4096,
                "temperature": 0.2,
            },
        },
    )

    answer = (
        res["candidates"][0]
        ["content"]
        ["parts"][0]
        ["text"]
        .strip()
    )
    # print("\n[FINAL ANSWER]")
    # print(answer)
    #
    # print("\n[FINAL SOURCES]")
    # print(found.sources)

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }

# =============================================================
# FastAPI
# ==============================================================
app = FastAPI()
inngest.fast_api.serve(app, inngest_client,[rag_ingest_pdf, rag_query_pdf_ai,],)