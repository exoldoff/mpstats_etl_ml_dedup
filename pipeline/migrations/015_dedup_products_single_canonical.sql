DELETE FROM mpstats_products_dedup
WHERE row_level = 'canonical';

INSERT INTO mpstats_products_dedup (
    run_id, project_name, category_key, category_name,
    ml_family_id, ml_pack_id, row_level, sort_order,
    node_id, canonical_node_id, canonical_sku, normalized_sku,
    marketplace_code, marketplace, article, sku, brand, subcategory,
    unit_amount, total_amount, multipack_count,
    sales_volume, revenue, source_row_count, component_size,
    ml_dedup_status, confidence_score
)
SELECT
    g.run_id,
    n.project_name,
    r.category_key,
    COALESCE(r.category_name, MIN(n.category_name)) AS category_name,
    g.ml_family_id,
    g.ml_pack_id,
    'canonical' AS row_level,
    0 AS sort_order,
    g.canonical_node_id AS node_id,
    g.canonical_node_id,
    MAX(g.canonical_sku) AS canonical_sku,
    MAX(g.canonical_sku) AS normalized_sku,
    MIN(cn.marketplace_code) AS marketplace_code,
    MIN(cn.marketplace) AS marketplace,
    MIN(cn.article) AS article,
    MIN(cn.sku) AS sku,
    MIN(cn.brand) AS brand,
    MIN(cn.subcategory) AS subcategory,
    MIN(cn.unit_amount) AS unit_amount,
    MIN(cn.total_amount) AS total_amount,
    MIN(cn.multipack_count) AS multipack_count,
    COALESCE(SUM(n.sales_volume), 0) AS sales_volume,
    COALESCE(SUM(n.revenue), 0) AS revenue,
    COALESCE(SUM(n.row_count), 0) AS source_row_count,
    COUNT(*) AS component_size,
    CASE WHEN COUNT(*) > 1 THEN 'canonical_group' ELSE 'canonical_singleton' END AS ml_dedup_status,
    MAX(g.confidence_score) AS confidence_score
FROM dedup_sku_groups AS g
JOIN dedup_sku_nodes AS n
  ON n.run_id = g.run_id AND n.node_id = g.node_id
JOIN dedup_runs AS r
  ON r.run_id = g.run_id
LEFT JOIN dedup_sku_nodes AS cn
  ON cn.run_id = g.run_id AND cn.node_id = g.canonical_node_id
WHERE g.run_id IN (
    SELECT DISTINCT run_id
    FROM mpstats_products_dedup
    WHERE row_level = 'member'
)
GROUP BY
    g.run_id,
    n.project_name,
    r.category_key,
    r.category_name,
    g.ml_family_id,
    g.ml_pack_id,
    g.canonical_node_id;
