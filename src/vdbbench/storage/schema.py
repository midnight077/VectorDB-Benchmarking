RESULTS_TABLE = "results"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    lane TEXT NOT NULL,
    dataset TEXT NOT NULL,
    backend TEXT NOT NULL,
    index_type TEXT NOT NULL,
    metric TEXT NOT NULL,
    dim INTEGER NOT NULL,
    build_params TEXT NOT NULL,
    search_params TEXT NOT NULL,
    k INTEGER NOT NULL,
    recall_at_k REAL NOT NULL,
    p50_ms REAL NOT NULL,
    p95_ms REAL NOT NULL,
    p99_ms REAL NOT NULL,
    mean_ms REAL NOT NULL,
    serial_qps REAL NOT NULL,
    build_time_s REAL NOT NULL,
    peak_rss_mb REAL NOT NULL,
    index_disk_mb REAL NOT NULL,
    n_queries INTEGER NOT NULL,
    n_warmup INTEGER NOT NULL,
    n_repeats INTEGER NOT NULL
)
"""
