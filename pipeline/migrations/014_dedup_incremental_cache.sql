CREATE TABLE IF NOT EXISTS dedup_node_embeddings (
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    node_id VARCHAR NOT NULL,
    embedding_model_name VARCHAR NOT NULL,
    text_builder_version VARCHAR NOT NULL,
    normalize_embeddings BOOLEAN NOT NULL,
    embedding_text_hash VARCHAR NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (
        project_name,
        category_key,
        node_id,
        embedding_model_name,
        text_builder_version,
        normalize_embeddings,
        embedding_text_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_dedup_node_embeddings_lookup
    ON dedup_node_embeddings(project_name, category_key, node_id, embedding_model_name);

CREATE TABLE IF NOT EXISTS dedup_pair_score_cache (
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    node_id_a VARCHAR NOT NULL,
    node_id_b VARCHAR NOT NULL,
    model_method VARCHAR NOT NULL,
    model_path VARCHAR,
    hf_model_id VARCHAR,
    activation VARCHAR,
    pair_text_hash VARCHAR NOT NULL,
    score DOUBLE NOT NULL,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (
        project_name,
        category_key,
        node_id_a,
        node_id_b,
        model_method,
        model_path,
        hf_model_id,
        activation,
        pair_text_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_dedup_pair_score_cache_lookup
    ON dedup_pair_score_cache(project_name, category_key, node_id_a, node_id_b);

CREATE TABLE IF NOT EXISTS dedup_identity_assignments (
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    node_id VARCHAR NOT NULL,
    model_method VARCHAR NOT NULL,
    model_path VARCHAR,
    hf_model_id VARCHAR,
    embedding_model_name VARCHAR,
    faiss_top_k INTEGER NOT NULL,
    activation VARCHAR,
    threshold_strategy VARCHAR NOT NULL,
    threshold_same DOUBLE NOT NULL,
    ml_family_id VARCHAR NOT NULL,
    ml_pack_id VARCHAR NOT NULL,
    canonical_node_id VARCHAR,
    canonical_sku VARCHAR,
    ml_dedup_status VARCHAR NOT NULL,
    confidence_score DOUBLE,
    component_size INTEGER DEFAULT 1,
    source_run_id VARCHAR,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (project_name, category_key, node_id)
);

CREATE INDEX IF NOT EXISTS idx_dedup_identity_assignments_group
    ON dedup_identity_assignments(project_name, category_key, ml_family_id, ml_pack_id);
