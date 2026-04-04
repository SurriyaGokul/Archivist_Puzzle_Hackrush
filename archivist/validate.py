from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from archivist.data import load_pages_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a submission CSV against a test CSV.")
    p.add_argument("--test_csv", required=True, help="Path to BookA_test.csv or BookB_test.csv")
    p.add_argument("--submission_csv", required=True, help="Path to BookA.csv or BookB.csv")
    return p.parse_args()


def validate_submission(test_csv: str | Path, submission_csv: str | Path) -> None:
    pages = load_pages_csv(test_csv)
    expected_ids = [p.page_id for p in pages]
    expected_set = set(expected_ids)
    n = len(expected_ids)

    df = pd.read_csv(submission_csv)
    if "original_page" not in df.columns or "shuffled_page" not in df.columns:
        raise ValueError("submission must have columns ['original_page','shuffled_page']")

    if len(df) != n:
        raise ValueError(f"submission has {len(df)} rows, expected {n}")

    orig = df["original_page"].tolist()
    if orig != list(range(1, n + 1)):
        raise ValueError("original_page must be 1..N in order")

    shuf = [int(x) for x in df["shuffled_page"].tolist()]
    if len(set(shuf)) != n:
        raise ValueError("shuffled_page contains duplicates")
    if set(shuf) != expected_set:
        missing = sorted(expected_set - set(shuf))
        extra = sorted(set(shuf) - expected_set)
        raise ValueError(f"shuffled_page mismatch: missing={missing[:10]} extra={extra[:10]}")


def main() -> None:
    args = _parse_args()
    validate_submission(args.test_csv, args.submission_csv)
    print("OK")


if __name__ == "__main__":
    main()
