-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk; (video_id, chunk_index) is the natural key.
-- Multiple chunks per video are stored as separate rows.
CREATE TABLE IF NOT EXISTS video_chunk_embeddings (
    id           SERIAL PRIMARY KEY,
    video_id     VARCHAR(255) NOT NULL,
    chunk_index  INTEGER      NOT NULL,
    start_time   FLOAT        NOT NULL,  -- seconds from video start
    end_time     FLOAT        NOT NULL,
    embedding    vector(768)  NOT NULL,
    created_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (video_id, chunk_index)
);

-- IVFFlat index for fast approximate cosine-similarity search
CREATE INDEX IF NOT EXISTS video_chunk_embeddings_embedding_idx
    ON video_chunk_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Index for the common WHERE clause pattern: video_id + start_time filter
CREATE INDEX IF NOT EXISTS video_chunk_embeddings_video_time_idx
    ON video_chunk_embeddings (video_id, start_time);
