CREATE TABLE IF NOT EXISTS dedup_runs (
    run_id VARCHAR PRIMARY KEY,
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    category_name VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'queued',
    model_method VARCHAR NOT NULL,
    model_path VARCHAR,
    hf_model_id VARCHAR,
    embedding_model_name VARCHAR,
    activation VARCHAR,
    threshold_strategy VARCHAR NOT NULL,
    threshold_same DOUBLE NOT NULL,
    faiss_top_k INTEGER NOT NULL,
    node_count INTEGER DEFAULT 0,
    candidate_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    group_count INTEGER DEFAULT 0,
    manifest_path VARCHAR,
    manifest_json VARCHAR,
    error VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dedup_sku_nodes (
    run_id VARCHAR NOT NULL,
    node_id VARCHAR NOT NULL,
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    category_name VARCHAR,
    marketplace_code VARCHAR,
    marketplace VARCHAR,
    article VARCHAR,
    sku VARCHAR,
    brand VARCHAR,
    subcategory VARCHAR,
    unit_amount DOUBLE,
    total_amount DOUBLE,
    multipack_count DOUBLE,
    sales_volume DOUBLE,
    revenue DOUBLE,
    row_count INTEGER DEFAULT 0,
    source_row_hashes_json VARCHAR,
    embedding_text VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (run_id, node_id)
);

CREATE TABLE IF NOT EXISTS dedup_sku_edges (
    run_id VARCHAR NOT NULL,
    edge_id VARCHAR NOT NULL,
    node_id_a VARCHAR NOT NULL,
    node_id_b VARCHAR NOT NULL,
    score DOUBLE,
    threshold_strategy VARCHAR NOT NULL,
    threshold_same DOUBLE NOT NULL,
    predicted_binary BOOLEAN NOT NULL,
    predicted_label VARCHAR NOT NULL,
    candidate_rank INTEGER,
    candidate_source VARCHAR,
    blocking_scope VARCHAR,
    same_pack_signature BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (run_id, edge_id)
);

CREATE TABLE IF NOT EXISTS dedup_sku_groups (
    run_id VARCHAR NOT NULL,
    node_id VARCHAR NOT NULL,
    ml_family_id VARCHAR NOT NULL,
    ml_pack_id VARCHAR NOT NULL,
    canonical_node_id VARCHAR,
    canonical_sku VARCHAR,
    ml_dedup_status VARCHAR NOT NULL,
    confidence_score DOUBLE,
    component_size INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (run_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_dedup_runs_project_category_status
    ON dedup_runs(project_name, category_key, status);

CREATE INDEX IF NOT EXISTS idx_dedup_nodes_lookup
    ON dedup_sku_nodes(run_id, project_name, category_key, marketplace_code, article);

CREATE INDEX IF NOT EXISTS idx_dedup_groups_lookup
    ON dedup_sku_groups(run_id, node_id);
