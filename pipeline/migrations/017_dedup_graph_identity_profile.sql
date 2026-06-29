ALTER TABLE dedup_identity_assignments
    ADD COLUMN IF NOT EXISTS graph_grouping_algorithm VARCHAR DEFAULT 'connected_components';

ALTER TABLE dedup_identity_assignments
    ADD COLUMN IF NOT EXISTS graph_community_resolution DOUBLE DEFAULT 1.0;

ALTER TABLE dedup_identity_assignments
    ADD COLUMN IF NOT EXISTS graph_community_seed INTEGER DEFAULT 42;

ALTER TABLE dedup_identity_assignments
    ADD COLUMN IF NOT EXISTS graph_edge_weight_col VARCHAR DEFAULT 'score';

UPDATE dedup_identity_assignments
SET
    graph_grouping_algorithm = COALESCE(NULLIF(graph_grouping_algorithm, ''), 'connected_components'),
    graph_community_resolution = COALESCE(graph_community_resolution, 1.0),
    graph_community_seed = COALESCE(graph_community_seed, 42),
    graph_edge_weight_col = COALESCE(NULLIF(graph_edge_weight_col, ''), 'score');
