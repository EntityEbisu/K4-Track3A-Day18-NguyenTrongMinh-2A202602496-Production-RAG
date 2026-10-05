from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split() for chunk in chunks
        ]

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        # Sắp xếp giảm dần rồi cắt top_k; doc không liên quan có score = 0 thì bỏ.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class _LMStudioEmbedder:
    """Bọc LM Studio OpenAI-compatible /v1/embeddings thành API .encode() giống
    SentenceTransformer, để DenseSearch không phải đổi logic gọi."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def encode(self, texts: str | list[str], **_ignored):
        from numpy import array, linalg

        if isinstance(texts, str):
            texts = [texts]

        from openai import OpenAI
        client = OpenAI(timeout=120.0, max_retries=2)

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            # Sắp xếp theo index để chắc chắn khớp thứ tự với batch input.
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))

        arr = array(vectors, dtype="float32")
        # LM Studio đã trả vector L2-normalised (norm = 1.0) nhưng normalize lại cho
        # chắc chắn, để cosine distance của Qdrant luôn đúng.
        norms = linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-9)


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        # Docker thuong khong chay trong lab → dùng Qdrant in-memory. Phai verify
        # collection list thuc su chay duoc (khong phai chi tao client) truoc khi giu
        # remote client, neu khong moi nham sang :memory:.
        self.client = QdrantClient(":memory:")
        try:
            remote = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2,
                                  check_compatibility=False)
            remote.get_collections()
            self.client = remote
            self.remote = True
        except Exception:
            self.remote = False
        self._encoder = None

    def _get_encoder(self):
        """Trả về object có .encode(texts) -> numpy array (L2-normalised, 1024-dim).

        Dùng LM Studio `/v1/embeddings` thay vì SentenceTransformer("BAAI/bge-m3"):
        cùng model bge-m3, nhưng đã được LM Studio nạp sẵn trên GPU nên không cần
        tải thêm ~2.3 GB. LM Studio phục vụ bản Q4_K_M GGUF nên vector khác
        bản full-precision một chút — không ảnh hưởng thứ hạng đáng kể.
        """
        if self._encoder is None:
            self._encoder = _LMStudioEmbedder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # create_collection raise nếu collection đã tồn tại → xoá trước cho idempotent.
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.DOT),
        )

        vectors = self._get_encoder().encode([c["text"] for c in chunks])
        self.client.upsert(collection, points=[
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self.client.collection_exists(collection):
            return []

        from qdrant_client import models

        query_vector = self._get_encoder().encode(query)[0].tolist()
        # NOTE 1/2/3 for qdrant-client 1.19 with QdrantClient(":memory:"):
        #   1) collection_name must be positional — collection= raises TypeError.
        #   2) a plain list is read as a 2-D array -> "Multivector  is not found";
        #      wrap it in models.NearestQuery(nearest=...).
        #   3) vectors are L2-normalised, so Distance.DOT == cosine.
        response = self.client.query_points(
            collection, query=models.NearestQuery(nearest=query_vector),
            limit=top_k)

        return [
            SearchResult(
                text=pt.payload["text"],
                score=float(pt.score),
                metadata=pt.payload,
                method="dense",
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, tuple[float, SearchResult]] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            total, representative = fused.get(result.text, (0.0, result))
            fused[result.text] = (total + 1.0 / (k + rank + 1), representative)

    ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)[:top_k]
    return [
        SearchResult(text=result.text, score=round(score, 6),
                     metadata=result.metadata, method="hybrid")
        for score, result in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
