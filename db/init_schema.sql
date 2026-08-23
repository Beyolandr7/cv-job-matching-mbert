-- ==============================================================================
-- DATABASE SCHEMA: JOB MATCHING SYSTEM (mBERT & CONTENT-BASED FILTERING)
-- Database Engine: PostgreSQL 14+
-- ==============================================================================

-- 1. EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==============================================================================
-- 2. TABEL: USERS
-- ==============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);


-- ==============================================================================
-- 3. TABEL: MATCHING_HISTORIES (RIWAYAT PENCOCOKAN CV & LOWONGAN)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS matching_histories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Input Dokumen & Lowongan dari Form
    cv_filename VARCHAR(255) NOT NULL,
    cv_text TEXT NOT NULL,
    job_title VARCHAR(255) NOT NULL,
    job_description TEXT NOT NULL,
    
    -- Metadata Entitas Hasil Ekstraksi NLP / NER (JSONB)
    extracted_entities JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Metrik Skor & Keputusan Sistem (mBERT Cosine Similarity)
    similarity_score NUMERIC(5, 4) NOT NULL CHECK (similarity_score >= 0.0000 AND similarity_score <= 1.0000),
    status_category VARCHAR(20) NOT NULL CHECK (status_category IN ('Fit', 'Close Fit', 'Not Fit')),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indeks Performa untuk Kueri Riwayat
CREATE INDEX IF NOT EXISTS idx_matching_histories_user_id ON matching_histories(user_id);
CREATE INDEX IF NOT EXISTS idx_matching_histories_created_at ON matching_histories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_matching_histories_status ON matching_histories(status_category);

-- GIN Index untuk pencarian cepat pada field JSONB
CREATE INDEX IF NOT EXISTS idx_matching_histories_entities_gin ON matching_histories USING GIN (extracted_entities);


-- ==============================================================================
-- 4. FUNCTION & TRIGGER: AUTO UPDATE 'updated_at' PADA USERS
-- ==============================================================================
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trg_update_users_timestamp ON users;
CREATE TRIGGER trg_update_users_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();