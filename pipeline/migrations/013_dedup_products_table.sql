CREATE TABLE IF NOT EXISTS mpstats_products_dedup (
    run_id VARCHAR NOT NULL,
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    category_name VARCHAR,
    ml_family_id VARCHAR NOT NULL,
    ml_pack_id VARCHAR NOT NULL,
    row_level VARCHAR NOT NULL,
    sort_order INTEGER NOT NULL,
    node_id VARCHAR,
    canonical_node_id VARCHAR,
    canonical_sku VARCHAR,
    normalized_sku VARCHAR,
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
    source_row_count INTEGER DEFAULT 0,
    component_size INTEGER DEFAULT 1,
    ml_dedup_status VARCHAR,
    confidence_score DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_mpstats_products_dedup_project_category
    ON mpstats_products_dedup(project_name, category_key);

CREATE INDEX IF NOT EXISTS idx_mpstats_products_dedup_run_level
    ON mpstats_products_dedup(run_id, row_level);
