CREATE TABLE IF NOT EXISTS dedup_manual_overrides (
    project_name VARCHAR NOT NULL,
    category_key VARCHAR NOT NULL,
    node_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    target_family_id VARCHAR,
    target_pack_id VARCHAR,
    source_run_id VARCHAR,
    note VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (project_name, category_key, node_id)
);

CREATE INDEX IF NOT EXISTS idx_dedup_manual_overrides_run
    ON dedup_manual_overrides(source_run_id);

CREATE INDEX IF NOT EXISTS idx_dedup_manual_overrides_action
    ON dedup_manual_overrides(project_name, category_key, action);
