"""CLI: turn data/raw/ -> data/processed/features.parquet."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_preprocessing",
        description="Apply the preprocessing Pipeline to all raw Trends pulls.",
    )
    p.add_argument("--config", default="config/predictors.yaml")
    p.add_argument("--out", default="data/processed/features.parquet")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "Phase 3 — see IMPLEMENTATION_PLAN.md §3 Phase 3. "
        f"Parsed args: {vars(args)}"
    )


if __name__ == "__main__":
    sys.exit(main())
