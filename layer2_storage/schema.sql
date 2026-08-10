-- Layer 2 schema. One PostgreSQL instance serves both stores.
--
-- Boundary rule: structured fields are queried directly with SQL and are NEVER
-- chunked or embedded. Only full OCR text is embedded, for retrieval only.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    dataset     TEXT NOT NULL,
    arm         TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    vendor      TEXT,
    doc_date    DATE,
    date_raw    TEXT,          -- as emitted, for auditing parse failures
    total       NUMERIC(12,2),
    address     TEXT,
    latency_ms  REAL,
    cost_usd    NUMERIC(10,6) DEFAULT 0,
    status      TEXT,
    raw         JSONB,
    PRIMARY KEY (dataset, arm, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_vendor ON documents (lower(vendor));
CREATE INDEX IF NOT EXISTS idx_documents_date   ON documents (doc_date);
CREATE INDEX IF NOT EXISTS idx_documents_total  ON documents (total);

-- local embeddings (all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    dataset   TEXT NOT NULL,
    doc_id    TEXT NOT NULL,
    chunk_no  INT  NOT NULL,
    strategy  TEXT NOT NULL,      -- whole | window | lines
    text      TEXT NOT NULL,
    embedding vector(384),
    UNIQUE (dataset, doc_id, chunk_no, strategy)
);

-- cloud embeddings (amazon.titan-embed-text-v2:0)
CREATE TABLE IF NOT EXISTS chunks_titan (
    id        BIGSERIAL PRIMARY KEY,
    dataset   TEXT NOT NULL,
    doc_id    TEXT NOT NULL,
    chunk_no  INT  NOT NULL,
    strategy  TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding vector(1024),
    UNIQUE (dataset, doc_id, chunk_no, strategy)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc       ON chunks (dataset, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_titan_doc ON chunks_titan (dataset, doc_id);

-- One HNSW index PER strategy. A single index over the whole table filters
-- AFTER the approximate scan, so the 50:1 lines:whole row ratio starved the
-- other strategies of results.
CREATE INDEX IF NOT EXISTS idx_chunks_vec_whole
    ON chunks USING hnsw (embedding vector_cosine_ops) WHERE strategy='whole';
CREATE INDEX IF NOT EXISTS idx_chunks_vec_window
    ON chunks USING hnsw (embedding vector_cosine_ops) WHERE strategy='window';
CREATE INDEX IF NOT EXISTS idx_chunks_vec_lines
    ON chunks USING hnsw (embedding vector_cosine_ops) WHERE strategy='lines';
CREATE INDEX IF NOT EXISTS idx_chunks_titan_vec_whole
    ON chunks_titan USING hnsw (embedding vector_cosine_ops) WHERE strategy='whole';
