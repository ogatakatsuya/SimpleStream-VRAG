from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from psycopg2.extensions import connection as PgConnection

from lib.encoder.clip_encoder import CLIPVideoEncoder, VideoChunk
from lib.memory.db.conn import get_connection
from lib.memory.db.operations import insert_chunk_embedding, search_similar_chunks


@dataclass
class SearchResult:
    video_id: str
    chunk_index: int
    start_time: float
    end_time: float
    distance: float  # cosine distance (lower = more similar)


class VideoRAG:
    """
    Video Retrieval-Augmented Generation pipeline backed by CLIP + pgvector.

    Usage
    -----
    rag = VideoRAG(chunk_duration=2.0)
    rag.index("video.mp4", fps=1.0)
    results = rag.query_by_text("a dog playing",
                                video_id="video",
                                query_time=30.0,
                                top_k=3)
    """

    def __init__(
        self,
        model_name: str = CLIPVideoEncoder.MODEL_NAME,
        chunk_duration: float = 1.0,
        device: str | None = None,
        encode_batch_size: int = 32,
        conn: PgConnection | None = None,
    ) -> None:
        self.encoder = CLIPVideoEncoder(
            model_name=model_name,
            device=device,
            encode_batch_size=encode_batch_size,
        )
        self.chunk_duration = chunk_duration
        self.conn = conn or get_connection()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        video_path: str | Path,
        fps: float,
        video_id: str | None = None,
    ) -> list[VideoChunk]:
        """
        Chunk `video_path` by `self.chunk_duration`, encode each chunk with
        CLIP, and upsert embeddings into PostgreSQL.

        Args:
            video_path: Path to the video file.
            fps: Target sampling rate in frames per second.
            video_id: Key stored in the DB. Defaults to the file stem.

        Returns:
            The list of VideoChunk objects that were indexed.
        """
        video_path = Path(video_path)
        video_id = video_id or video_path.stem

        chunks = self.encoder.encode_video(
            video_path, fps=fps, chunk_duration=self.chunk_duration
        )
        for chunk in chunks:
            insert_chunk_embedding(
                self.conn,
                video_id=video_id,
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                embedding=chunk.embedding,
            )
        return chunks

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_by_text(
        self,
        text: str,
        video_id: str,
        query_time: float,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieve chunks most similar to a text query, restricted to chunks
        from `video_id` that start before `query_time`.
        """
        query_emb = self.encoder.encode_text(text)
        rows = search_similar_chunks(
            self.conn,
            query_embedding=query_emb,
            video_id=video_id,
            query_time=query_time,
            limit=top_k,
        )
        return [SearchResult(*row) for row in rows]

    def query_by_image(
        self,
        image: str | Path | Image.Image,
        video_id: str,
        query_time: float,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieve chunks most similar to a query image, restricted to chunks
        from `video_id` that start before `query_time`.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        query_emb = self.encoder.encode_frames([image])
        rows = search_similar_chunks(
            self.conn,
            query_embedding=query_emb,
            video_id=video_id,
            query_time=query_time,
            limit=top_k,
        )
        return [SearchResult(*row) for row in rows]
