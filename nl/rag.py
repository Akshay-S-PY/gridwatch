"""
Semantic retrieval over historical anomalies (the RAG piece).

Anomaly rows are turned into short text summaries, embedded (OpenAI) and stored in
Qdrant. At query time we embed the operator's question and pull the most similar
past anomalies to ground the answer ("which past events look like this?").
"""
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sqlalchemy import text

from db.config import engine
from llm import client as llm

logger = logging.getLogger(__name__)

COLLECTION = "anomaly_summaries"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def _client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _summary(row: dict) -> str:
    return (
        f"{row['timestamp']:%Y-%m-%d %H:%M UTC}: {row['severity']} anomaly in "
        f"{row['signal']} — value {row['value']:.1f} (isolation score {row['anomaly_score']:.2f})."
    )


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=qm.VectorParams(size=llm.EMBED_DIM, distance=qm.Distance.COSINE),
        )
        logger.info(f"rag: created Qdrant collection '{COLLECTION}'")


def seed(force: bool = False) -> int:
    """
    Embed all anomaly_flags summaries into Qdrant. Skips if already populated
    (unless force=True). Returns the number of points indexed.
    """
    client = _client()
    ensure_collection(client)

    if not force:
        existing = client.count(COLLECTION, exact=True).count
        if existing > 0:
            logger.info(f"rag: collection already has {existing} points — skipping seed")
            return 0

    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(
            "SELECT id, timestamp, signal, value, anomaly_score, severity "
            "FROM anomaly_flags ORDER BY timestamp"
        )).mappings()]
    if not rows:
        logger.info("rag: no anomalies to seed")
        return 0

    summaries = [_summary(r) for r in rows]
    # OpenAI embeds the whole batch in one call.
    vectors = llm.embed(summaries)
    points = [
        qm.PointStruct(
            id=r["id"],
            vector=vec,
            payload={
                "timestamp": r["timestamp"].isoformat(),
                "signal": r["signal"], "value": r["value"],
                "severity": r["severity"], "summary": s,
            },
        )
        for r, s, vec in zip(rows, summaries, vectors)
    ]
    client.upsert(COLLECTION, points=points)
    logger.info(f"rag: indexed {len(points)} anomaly summaries")
    return len(points)


def search(question: str, k: int = 5) -> list[str]:
    """Return the summaries of the k most similar past anomalies (empty on failure)."""
    try:
        client = _client()
        if not client.collection_exists(COLLECTION):
            return []
        qvec = llm.embed([question])[0]
        hits = client.search(collection_name=COLLECTION, query_vector=qvec, limit=k)
        return [h.payload["summary"] for h in hits]
    except Exception as e:  # noqa: BLE001 — RAG is best-effort context, never fatal
        logger.warning(f"rag: search failed: {e}")
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed()
