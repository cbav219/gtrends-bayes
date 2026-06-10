"""CLI: fit BSTS for one or more targets and persist the posterior."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fit_bsts",
        description="Fit a BSTS model and save its posterior draws.",
    )
    p.add_argument("--target", choices=["HY_OAS", "IG_OAS", "BOTH"], default="BOTH")
    p.add_argument("--features", default="data/processed/features.parquet")
    p.add_argument("--model-config", default="config/model.yaml")
    p.add_argument("--out-dir", default="data/processed/posterior")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "Phase 5 — see IMPLEMENTATION_PLAN.md §3 Phase 5. "
        f"Parsed args: {vars(args)}"
    )


if __name__ == "__main__":
    sys.exit(main())
