from __future__ import annotations

import numpy as np
from psycopg2.extensions import connection


def insert_chunk_embedding(
    conn: connection,
    video_id: str,
    chunk_index: int,
    start_time: float,
    end_time: float,
    embedding: np.ndarray,
) -> None:
    """
    Upsert a single chunk embedding.  Multiple chunks per video_id are stored
    as separate rows distinguished by chunk_index.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_chunk_embeddings
                (video_id, chunk_index, start_time, end_time, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (video_id, chunk_index) DO UPDATE
                SET start_time = EXCLUDED.start_time,
                    end_time   = EXCLUDED.end_time,
                    embedding  = EXCLUDED.embedding
            """,
            (
                video_id,
                chunk_index,
                start_time,
                end_time,
                "[" + ",".join(map(str, embedding.tolist())) + "]",
            ),
        )
    conn.commit()


def search_similar_chunks(
    conn: connection,
    query_embedding: np.ndarray,
    video_id: str,
    query_time: float,
    limit: int = 5,
) -> list[tuple[str, int, float, float, float]]:
    """
    Find the most similar chunks within the same video that start before query_time.

    Args:
        conn: PostgreSQL connection.
        query_embedding: L2-normalized query vector (768,).
        video_id: Restrict search to this video.
        query_time: Only consider chunks with start_time < query_time.
        limit: Number of results to return.

    Returns:
        List of (video_id, chunk_index, start_time, end_time, distance).
        Ordered by ascending cosine distance (most similar first).
    """
    embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                video_id,
                chunk_index,
                start_time,
                end_time,
                embedding <=> %s::vector AS distance
            FROM video_chunk_embeddings
            WHERE video_id = %s
              AND start_time < %s
            ORDER BY distance
            LIMIT %s
            """,
            (embedding_str, video_id, query_time, limit),
        )
        return cur.fetchall()
