-- Si-Periksa v2.0 — SQLite WAL Schema
-- Zero-PII: tidak ada data mikro individu di sini, hanya metadata & status

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Registri buklet (Mendukung multi-survei: seruti_inti, seruti_mak, susenas_kp)
CREATE TABLE IF NOT EXISTS booklets (
    id          TEXT PRIMARY KEY,
    survey_type TEXT DEFAULT 'seruti_inti', -- 'seruti_inti' | 'seruti_mak' | 'susenas_kp'
    nks         TEXT,
    nus         TEXT,
    nama_krt    TEXT,
    scan_path   TEXT,
    gate_status TEXT DEFAULT 'PENDING',
    gate_detail TEXT DEFAULT '{}',
    ocr_status  TEXT DEFAULT 'PENDING',
    hitl_count  INTEGER DEFAULT 0,
    hitl_done   INTEGER DEFAULT 0,
    val_status  TEXT DEFAULT 'PENDING',
    page_count  INTEGER DEFAULT 7,
    dsrt_linked INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    updated_at  TEXT DEFAULT (datetime('now','localtime'))
);

-- Master Prelist DSRT (Daftar Sampel Rumah Tangga)
CREATE TABLE IF NOT EXISTS dsrt_files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_type   TEXT DEFAULT 'seruti_inti',
    filename      TEXT NOT NULL,
    file_path     TEXT,
    total_rows    INTEGER DEFAULT 0,
    matched_count INTEGER DEFAULT 0,
    uploaded_at   TEXT DEFAULT (datetime('now','localtime')),
    is_active     INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS dsrt_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER REFERENCES dsrt_files(id),
    survey_type   TEXT DEFAULT 'seruti_inti',
    nks           TEXT,
    nus           TEXT,
    nama_krt      TEXT,
    alamat        TEXT,
    jart          INTEGER,
    status_rt     TEXT DEFAULT 'Utama',
    no_bf         TEXT,
    no_bs         TEXT,
    booklet_id    TEXT,
    is_matched    INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_dsrt_survey ON dsrt_samples(survey_type);
CREATE INDEX IF NOT EXISTS idx_dsrt_nks_nus ON dsrt_samples(nks, nus);
CREATE INDEX IF NOT EXISTS idx_dsrt_booklet ON dsrt_samples(booklet_id);

-- Antrean HITL
CREATE TABLE IF NOT EXISTS hitl_queue (
    id           TEXT PRIMARY KEY,
    booklet_id   TEXT REFERENCES booklets(id),
    variable     TEXT NOT NULL,
    label        TEXT,
    block_name   TEXT,
    page         INTEGER,
    pred_digit   TEXT,
    pred_conf    REAL,
    top3_json    TEXT DEFAULT '[]',
    hint         TEXT DEFAULT '',
    corrected    INTEGER DEFAULT 0,
    final_digit  TEXT,
    corrected_at TEXT,
    tier         INTEGER DEFAULT 7,
    crop_path    TEXT,
    priority     INTEGER DEFAULT 0
);

-- Audit trail koreksi
CREATE TABLE IF NOT EXISTS correction_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    hitl_id      TEXT REFERENCES hitl_queue(id),
    booklet_id   TEXT,
    variable     TEXT,
    old_digit    TEXT,
    new_digit    TEXT,
    operator     TEXT DEFAULT 'operator_01',
    corrected_at TEXT DEFAULT (datetime('now','localtime'))
);

-- Konfigurasi sistem
CREATE TABLE IF NOT EXISTS system_config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

INSERT OR IGNORE INTO system_config VALUES ('active_survey', 'seruti_inti');
INSERT OR IGNORE INTO system_config VALUES ('confidence_threshold', '85');
INSERT OR IGNORE INTO system_config VALUES ('auto_hot_folder', 'true');
INSERT OR IGNORE INTO system_config VALUES ('auto_reconcile', 'true');
INSERT OR IGNORE INTO system_config VALUES ('strict_zero_pii', 'true');
INSERT OR IGNORE INTO system_config VALUES ('lan_mode', 'false');
INSERT OR IGNORE INTO system_config VALUES ('model_version', 'v3.2');
INSERT OR IGNORE INTO system_config VALUES ('worker_count', '4');

-- Pipeline status
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_type  TEXT DEFAULT 'seruti_inti',
    started_at   TEXT DEFAULT (datetime('now','localtime')),
    finished_at  TEXT,
    status       TEXT DEFAULT 'RUNNING',
    booklets_total INTEGER DEFAULT 0,
    booklets_done  INTEGER DEFAULT 0,
    summary_json   TEXT DEFAULT '{}'
);

-- Hasil Inferensi AI per Buklet (Penyimpanan Dokumen Ekstraksi Lengkap)
CREATE TABLE IF NOT EXISTS booklet_inferences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    booklet_id       TEXT NOT NULL REFERENCES booklets(id),
    model_name       TEXT DEFAULT 'gemini_38_flash',
    total_digits     INTEGER DEFAULT 0,
    blank_digits     INTEGER DEFAULT 0,
    filled_digits    INTEGER DEFAULT 0,
    auto_accepted    INTEGER DEFAULT 0,
    needs_review     INTEGER DEFAULT 0,
    avg_confidence   REAL DEFAULT 0.0,
    extraction_json  TEXT DEFAULT '{}',
    validation_json  TEXT DEFAULT '{}',
    created_at       TEXT DEFAULT (datetime('now','localtime')),
    updated_at       TEXT DEFAULT (datetime('now','localtime')),
    is_latest        INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_inferences_booklet ON booklet_inferences(booklet_id);
CREATE INDEX IF NOT EXISTS idx_inferences_latest ON booklet_inferences(booklet_id, is_latest);

-- Nilai Variabel Hasil Inferensi AI per Buklet (Akses Granular Kolom & Baris)
CREATE TABLE IF NOT EXISTS booklet_field_values (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    booklet_id   TEXT NOT NULL REFERENCES booklets(id),
    page         INTEGER,
    block_name   TEXT,
    field_id     TEXT NOT NULL,
    var_name     TEXT NOT NULL,
    var_label    TEXT,
    data_type    TEXT,
    raw_string   TEXT,
    value        TEXT,
    confidence   REAL DEFAULT 1.0,
    needs_review INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now','localtime')),
    updated_at   TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(booklet_id, var_name)
);

CREATE INDEX IF NOT EXISTS idx_field_values_booklet ON booklet_field_values(booklet_id);
CREATE INDEX IF NOT EXISTS idx_field_values_var ON booklet_field_values(var_name);

-- Index
CREATE INDEX IF NOT EXISTS idx_booklets_survey ON booklets(survey_type);
CREATE INDEX IF NOT EXISTS idx_hitl_booklet ON hitl_queue(booklet_id);
CREATE INDEX IF NOT EXISTS idx_hitl_corrected ON hitl_queue(corrected);
CREATE INDEX IF NOT EXISTS idx_hitl_priority ON hitl_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_correction_log_hitl ON correction_log(hitl_id);
