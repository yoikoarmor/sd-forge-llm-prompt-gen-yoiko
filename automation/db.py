"""SQLite storage for automation runs and scores."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    gen_prompt TEXT NOT NULL,
    parts_json TEXT,
    final_prompt TEXT,
    negative_prompt TEXT,
    llm_mode TEXT,
    seed INTEGER,
    image_path TEXT,
    info_json TEXT,
    clip_score REAL,
    clip_emb BLOB,
    vlm_json TEXT,
    vlm_overall REAL,
    disc_score REAL
);
"""


def connect_db(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_run(
    conn,
    created_at,
    gen_prompt,
    parts_json,
    final_prompt,
    negative_prompt,
    llm_mode,
    seed,
    image_path,
    info_json,
):
    cur = conn.execute(
        """
        INSERT INTO runs (
            created_at, gen_prompt, parts_json, final_prompt,
            negative_prompt, llm_mode, seed, image_path, info_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            gen_prompt,
            parts_json,
            final_prompt,
            negative_prompt,
            llm_mode,
            seed,
            image_path,
            info_json,
        ),
    )
    conn.commit()
    return cur.lastrowid


def rows_missing_clip(conn):
    return conn.execute(
        "SELECT id, gen_prompt, image_path FROM runs "
        "WHERE clip_score IS NULL AND image_path IS NOT NULL"
    ).fetchall()


def rows_missing_vlm(conn):
    return conn.execute(
        "SELECT id, gen_prompt, image_path FROM runs "
        "WHERE vlm_overall IS NULL AND image_path IS NOT NULL"
    ).fetchall()


def rows_missing_disc(conn):
    return conn.execute(
        "SELECT id, clip_emb FROM runs "
        "WHERE disc_score IS NULL AND clip_emb IS NOT NULL"
    ).fetchall()


def update_clip(conn, run_id, clip_score, clip_emb_bytes):
    conn.execute(
        "UPDATE runs SET clip_score = ?, clip_emb = ? WHERE id = ?",
        (clip_score, clip_emb_bytes, run_id),
    )
    conn.commit()


def update_vlm(conn, run_id, vlm_json, vlm_overall):
    conn.execute(
        "UPDATE runs SET vlm_json = ?, vlm_overall = ? WHERE id = ?",
        (vlm_json, vlm_overall, run_id),
    )
    conn.commit()


def update_disc(conn, run_id, disc_score):
    conn.execute(
        "UPDATE runs SET disc_score = ? WHERE id = ?",
        (disc_score, run_id),
    )
    conn.commit()
