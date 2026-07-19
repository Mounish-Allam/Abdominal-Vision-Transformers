-- Documented SQL queries over outputs/analytics.db (populated by evaluate.py and
-- src/demo.py via analytics.py). Run with: sqlite3 outputs/analytics.db < queries.sql
-- or open the file directly in any SQLite client.

-- 1. Mean per-organ Dice per model, from evaluate.py runs only (source='evaluate',
--    the only source with ground truth). This is the same aggregate
--    scripts/make_results_table.py computes from the JSON outputs, in SQL instead
--    of pandas - a real cross-check that both paths agree.
SELECT
    i.model_name,
    o.organ,
    ROUND(AVG(o.dice), 4)  AS mean_dice,
    COUNT(*)               AS n_slices
FROM organ_stats o
JOIN inferences i ON i.id = o.inference_id
WHERE i.source = 'evaluate' AND o.dice IS NOT NULL
GROUP BY i.model_name, o.organ
ORDER BY i.model_name, mean_dice DESC;

-- 2. Mean confidence and entropy per organ (SwinDAF only).
SELECT
    o.organ,
    ROUND(AVG(o.mean_confidence), 4) AS avg_confidence,
    ROUND(AVG(o.mean_entropy), 4)    AS avg_entropy,
    COUNT(*)                         AS n_slices
FROM organ_stats o
JOIN inferences i ON i.id = o.inference_id
WHERE i.model_name = 'swin_daf' AND o.pixels > 0
GROUP BY o.organ
ORDER BY avg_confidence ASC;

-- 3. Count of low-confidence organ predictions per organ (SwinDAF), i.e. how many
--    slices would trigger the "requires human review" flag in the clinical report.
SELECT
    o.organ,
    SUM(o.low_confidence)  AS low_confidence_count,
    COUNT(*)                AS total_slices,
    ROUND(100.0 * SUM(o.low_confidence) / COUNT(*), 1) AS pct_flagged
FROM organ_stats o
JOIN inferences i ON i.id = o.inference_id
WHERE i.model_name = 'swin_daf'
GROUP BY o.organ
ORDER BY pct_flagged DESC;

-- 4. Mean inference latency per model (segmentation forward pass only).
SELECT
    model_name,
    ROUND(AVG(latency_seconds), 4) AS avg_latency_seconds,
    COUNT(*)                       AS n_inferences
FROM inferences
GROUP BY model_name;

-- 5. Worst-performing individual slices for SwinDAF (any organ with real ground
--    truth scoring below 0.1 Dice) - a SQL-native way to find the Subject 15
--    failure cases documented in the README's Failure analysis section.
SELECT
    i.slice_id,
    o.organ,
    ROUND(o.dice, 4) AS dice
FROM organ_stats o
JOIN inferences i ON i.id = o.inference_id
WHERE i.model_name = 'swin_daf' AND o.dice IS NOT NULL AND o.dice < 0.1
ORDER BY o.dice ASC
LIMIT 10;
