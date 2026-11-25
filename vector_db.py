from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams, SparseVectorParams, SparseVector,
    Distance, Modifier, PointStruct, Prefetch, RrfQuery, Rrf, MatchAny,
    Filter, FieldCondition, MatchValue,)
from fastembed import SparseTextEmbedding

class QdrantStorage:

    def __init__(self, url="http://localhost:6333", collection="docs_hybrid_qwen", dim=2560,):
        self.client = QdrantClient(url=url, timeout=30,)
        self.collection = collection
        # Local BM25 model
        self.bm25 = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )
        # Create collection only once
        if not self.client.collection_exists(self.collection):

            self.client.create_collection(
                collection_name=self.collection,
                # Dense qwen embeddings
                vectors_config={
                    "dense": VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    )
                },

                # BM25 sparse vectors
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        modifier=Modifier.IDF
                    )
                },
            )

    # ----------------------------------------------------------
    # Store Dense + BM25
    # ----------------------------------------------------------

    def upsert_hybrid(self, ids, dense_vectors, texts, payloads,):

        # Convert every chunk into a BM25 sparse vector
        sparse_vectors = list(
            self.bm25.embed(texts)
        )

        points = []

        for i in range(len(ids)):
            sparse = sparse_vectors[i]
            point = PointStruct(
                id=ids[i],
                vector={
                    # Gemini dense vector
                    "dense": dense_vectors[i],

                    # BM25 sparse vector
                    "bm25": SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },

                payload=payloads[i],
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    # ----------------------------------------------------------
    # Hybrid Search = Dense + BM25 + RRF
    # ----------------------------------------------------------

    def hybrid_search(self, query_vector, query_text, source_ids: list[str], top_k: int = 5,):
        # BM25 representation of the question
        sparse_query = next(self.bm25.query_embed(query_text))
        candidate_limit = max(top_k * 4, 20)

        # Only allow chunks belonging to this PDF
        source_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchAny(
                        any=source_ids
                    ),
                )
            ]
        )

        response = self.client.query_points(
            collection_name=self.collection,

            prefetch=[
                # Dense search
                Prefetch(query=query_vector, using="dense", limit=candidate_limit,),

                # BM25 search
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="bm25",
                    limit=candidate_limit,
                ),
            ],

            query=RrfQuery(
                rrf=Rrf()
            ),
            query_filter=source_filter,
            with_payload=True,
            limit=top_k,
        )

        contexts = []
        sources = set()

        for result in response.points:

            payload = result.payload or {}

            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }