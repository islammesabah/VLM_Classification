"""
Build the merged GTSRB analysis database.

GPT-4o's original GTSRB run stored a free-text answer instead of the JSON
traversal path for 667 of its 901 primary-condition images, so only 234
traversals (25 of 43 classes) were usable for node-level analysis -- and the
usable subset was biased, scoring 31.6% tree accuracy against 66.7% for the
rest. GPT-4o was therefore re-run over all 901 images with path logging
intact (results/gtsrb_gpt4o_paths.db); the re-run reproduces the original
tree accuracy (57.71% vs 57.60%) while yielding usable paths for 901/901
images across all 43 classes.

This script composes the two sources without modifying or deleting either:

  * every row of results/gtsrb.db EXCEPT GPT-4o's primary condition
  * GPT-4o's primary-condition rows from results/gtsrb_gpt4o_paths.db

Each resulting row therefore comes from exactly one run; paths and class
predictions are never mixed within a row. GPT-4o's non-primary GTSRB
configurations (other temperatures, prompt variants, RunIds) are untouched,
since the re-run covers only the primary condition.

Usage:
    python results/build_gtsrb_merged.py
"""
import shutil
import sqlite3
from pathlib import Path

SOURCE = "results/gtsrb.db"
RERUN = "results/gtsrb_gpt4o_paths.db"
MERGED = "results/gtsrb_merged.db"

PRIMARY = ("LLM_name = 'gpt-4o' AND temperature = 0.7 AND Include_memory = '0' "
           "AND Include_description = '0' AND RunId = 0")


def main():
    for p in (SOURCE, RERUN):
        if not Path(p).exists():
            raise SystemExit(f"missing input: {p}")

    if Path(MERGED).exists():
        Path(MERGED).unlink()

    # Start from the schema only, then populate by construction.
    src = sqlite3.connect(SOURCE)
    schema = src.execute(
        "SELECT sql FROM sqlite_master WHERE name='answers'").fetchone()[0]
    src.close()

    out = sqlite3.connect(MERGED)
    out.execute(schema)
    out.commit()

    out.execute(f"ATTACH DATABASE '{SOURCE}' AS src")
    out.execute(f"ATTACH DATABASE '{RERUN}' AS rerun")

    kept = out.execute(
        f"INSERT INTO answers SELECT * FROM src.answers WHERE NOT ({PRIMARY})").rowcount
    added = out.execute(
        f"INSERT INTO answers SELECT * FROM rerun.answers WHERE {PRIMARY}").rowcount
    out.commit()

    total = out.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    print(f"carried over from {SOURCE}: {kept}")
    print(f"taken from {RERUN}:      {added}")
    print(f"merged total:                {total} -> {MERGED}")
    out.close()


if __name__ == "__main__":
    main()
