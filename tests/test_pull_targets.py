"""Tests for scripts.pull_targets helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to import path so we can test the script's helper directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pull_targets  # noqa: E402


def test_formula_dependencies_extracts_fred_tickers():
    deps = pull_targets._formula_dependencies("DGS10 - DGS2")
    assert deps == ["DGS10", "DGS2"]


def test_formula_dependencies_ignores_lowercase_words():
    deps = pull_targets._formula_dependencies("max(DGS10, 0) - DGS2")
    assert deps == ["DGS10", "DGS2"]


def test_formula_dependencies_drops_single_letter_tokens():
    deps = pull_targets._formula_dependencies("A + B + DGS10")
    assert deps == ["DGS10"]


def test_build_parser_help_runs():
    """Smoke: --help should not raise."""
    parser = pull_targets.build_parser()
    assert parser.prog == "pull_targets"


def test_targets_config_has_etf_sources():
    """The committed targets.yaml should reference yfinance for HY and IG."""
    from gtrends_bayes.config import TargetsConfig

    cfg = TargetsConfig.from_yaml(Path(__file__).resolve().parents[1] / "config/targets.yaml")
    by_name = {t.name: t for t in cfg.targets}
    assert by_name["HY"].source == "yfinance" and by_name["HY"].ticker == "HYG"
    assert by_name["IG"].source == "yfinance" and by_name["IG"].ticker == "LQD"
