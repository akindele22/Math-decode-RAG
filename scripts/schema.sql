-- Math & Quant AI — MySQL metadata schema
-- Run once to set up the database

CREATE DATABASE IF NOT EXISTS math_quant_ai;
USE math_quant_ai;

-- Track every ingested document
CREATE TABLE IF NOT EXISTS documents (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    filename      VARCHAR(255)  NOT NULL,
    source_path   TEXT,
    total_pages   INT           DEFAULT 0,
    total_chunks  INT           DEFAULT 0,
    ingested_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    status        ENUM('pending','ingested','failed') DEFAULT 'ingested',
    notes         TEXT
);

-- Track individual query history
CREATE TABLE IF NOT EXISTS query_log (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    query_text    TEXT          NOT NULL,
    top_k         INT           DEFAULT 5,
    llm_provider  VARCHAR(50),
    answer_length INT,
    sources_used  JSON,
    created_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX idx_documents_filename ON documents(filename);
CREATE INDEX idx_querylog_created   ON query_log(created_at);
