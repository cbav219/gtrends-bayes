"""CLI: pack the v4 deliverable bundle from a clean repo state.

Produces a single tarball at ``dist/v4/gtrends-bayes-v4.tar.gz`` containing:

    gtrends-bayes-v4/
    ├── README.md / USAGE.md / BOOTSTRAP.md
    ├── requirements.txt          (minimum-viable deps; NO R / rpy2)
    ├── src/gtrends_bayes/        (inference/ + minimal supporting modules)
    ├── model/HY_v4.pkl + IG_v4.pkl
    ├── data/README.md            (sideband contract — no real data files)
    ├── scripts/verify_data.py + example_forecast.py
    └── MANIFEST.json             (SHA256 of every file)

Hard rules (validated before tarballing — script aborts on violation):
  - No R / rpy2 imports anywhere in the bundled src/.
  - No data/raw*/ or data/processed/ files.
  - Tarball ≤ 8 MB compressed (target ≤ 4 MB).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

# Minimum-viable supporting modules — copied into the bundle alongside
# inference/. These don't import R or rpy2.
_BUNDLED_SUPPORT_MODULES: tuple[str, ...] = (
    "__init__.py",
    "logging.py",
    "preprocessing/__init__.py",
    "preprocessing/target_transform.py",
)

# Minimum-viable runtime deps; the VM kernel will already have most of these.
_REQUIREMENTS_TXT = """\
pandas>=2.2
numpy>=1.26
pyarrow>=16
pyyaml>=6.0
"""

# Module name prefixes (matched against real ``import`` / ``from``
# statements via AST) that would pull R / rpy2 — or a deferred-stub like
# the WRDS-only OAS fetcher — into the bundled inference layer. Abort
# the build if any bundled .py file imports one of these.
_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "rpy2",
    "wrds",                               # WRDS client (used only by data/oas.py)
    "fredapi",                            # FRED client (training-time only)
    "gtrends_bayes.models.bsts",          # R bridge to bsts
    "gtrends_bayes.models.bsts_r",
    "gtrends_bayes.data.oas",             # deferred per v3 Phase A
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build_v4_bundle")
    p.add_argument("--bundle-version", default="v4",
                   help="Bundle version tag (e.g. 'v4', 'v5'). Drives the "
                        "inner directory name, the tarball filename suffix, "
                        "and the model-pickle glob. Default 'v4'.")
    p.add_argument("--model-dir", default=None,
                   help="Where the frozen *_{version}.pkl files live (Phase 1 "
                        "output). Defaults to dist/{bundle-version}/model.")
    p.add_argument("--docs-dir", default="docs/v4",
                   help="Where the README / USAGE / BOOTSTRAP markdown live.")
    p.add_argument("--out", default=None,
                   help="Output tarball path. Defaults to "
                        "dist/{bundle-version}/gtrends-bayes-{bundle-version}.tar.gz.")
    p.add_argument("--max-mb", type=float, default=8.0)
    p.add_argument("--max-pickle-mb", type=float, default=2.0)
    return p


def _imported_modules(py_file: Path) -> set[str]:
    """Parse ``py_file`` and return the set of imported module names.

    Uses AST so docstring / comment mentions don't false-positive.
    """
    try:
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
    except SyntaxError as exc:
        raise RuntimeError(f"{py_file} doesn't parse: {exc}") from exc
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


def _check_no_r_imports(py_file: Path) -> list[str]:
    """Return any forbidden module prefix imported by ``py_file``."""
    imports = _imported_modules(py_file)
    bad: list[str] = []
    for mod in imports:
        for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
            if mod == forbidden or mod.startswith(forbidden + "."):
                bad.append(mod)
                break
    return bad


def _check_inference_imports_clean(inf_dir: Path) -> None:
    """Walk the inference/ tree and verify no R / rpy2 imports."""
    for py in inf_dir.rglob("*.py"):
        bad = _check_no_r_imports(py)
        if bad:
            raise RuntimeError(
                f"{py} imports forbidden modules {bad}; "
                "the v4 bundle must be pure Python (no R / rpy2)."
            )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_into_staging(repo_root: Path, staging: Path,
                       model_dir: Path, docs_dir: Path,
                       max_pickle_mb: float, bundle_version: str) -> None:
    """Lay out the bundle contents under ``staging/gtrends-bayes-{ver}/``."""
    pkg_root = staging / f"gtrends-bayes-{bundle_version}"
    pkg_root.mkdir(parents=True)

    # --- inference/ + minimal supporting modules ---
    src_dst = pkg_root / "src" / "gtrends_bayes"
    src_dst.mkdir(parents=True)
    # Copy inference/ wholesale, skipping __pycache__ / .pytest_cache.
    inf_src = repo_root / "src" / "gtrends_bayes" / "inference"
    shutil.copytree(
        inf_src, src_dst / "inference",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )
    _check_inference_imports_clean(src_dst / "inference")
    # Copy listed supporting modules.
    for rel in _BUNDLED_SUPPORT_MODULES:
        s = repo_root / "src" / "gtrends_bayes" / rel
        d = src_dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(s, d)
        bad = _check_no_r_imports(d)
        if bad:
            raise RuntimeError(
                f"supporting module {rel} contains forbidden tokens {bad}"
            )

    # --- model/ ---
    model_dst = pkg_root / "model"
    model_dst.mkdir()
    pickles = sorted(Path(model_dir).glob(f"*_{bundle_version}.pkl"))
    if not pickles:
        raise RuntimeError(
            f"no *_{bundle_version}.pkl files in {model_dir}; run "
            f"scripts/freeze_model_v4.py --bundle-version {bundle_version} first"
        )
    for pkl in pickles:
        size_mb = pkl.stat().st_size / 1024 / 1024
        if size_mb > max_pickle_mb:
            raise RuntimeError(
                f"{pkl} is {size_mb:.2f} MB > {max_pickle_mb} MB pickle cap. "
                "Strip more posterior summary."
            )
        shutil.copy(pkl, model_dst / pkl.name)

    # --- scripts ---
    scripts_dst = pkg_root / "scripts"
    scripts_dst.mkdir()
    for name in ("verify_data.py", "example_forecast.py"):
        shutil.copy(repo_root / "scripts" / name, scripts_dst / name)

    # --- docs ---
    docs_src = Path(docs_dir)
    if docs_src.exists():
        for name in ("README.md", "USAGE.md", "BOOTSTRAP.md"):
            src = docs_src / name
            if src.exists():
                shutil.copy(src, pkg_root / name)
        # data sideband doc
        data_dst = pkg_root / "data"
        data_dst.mkdir()
        data_readme = docs_src / "data_README.md"
        if data_readme.exists():
            shutil.copy(data_readme, data_dst / "README.md")
    else:
        log.warning("docs dir %s not present; bundle will lack README/USAGE/BOOTSTRAP",
                    docs_src)

    # --- requirements.txt ---
    (pkg_root / "requirements.txt").write_text(_REQUIREMENTS_TXT)


def _verify_bundle_contents(pkg_root: Path) -> None:
    """Walk staged bundle; refuse anything from data/raw* or data/processed/."""
    for f in pkg_root.rglob("*"):
        rel = f.relative_to(pkg_root).as_posix()
        if rel.startswith("data/raw") or "data/processed" in rel:
            raise RuntimeError(
                f"bundle contains forbidden data path: {rel}. "
                "raw and processed data ship separately as the manual sideband."
            )


def _build_manifest(pkg_root: Path, args: argparse.Namespace,
                    bundle_version: str) -> dict:
    files = []
    for f in sorted(pkg_root.rglob("*")):
        if f.is_file():
            rel = f.relative_to(pkg_root).as_posix()
            files.append({
                "path": rel,
                "size_bytes": f.stat().st_size,
                "sha256": _sha256(f),
            })
    return {
        "name": f"gtrends-bayes-{bundle_version}",
        "bundle_version": bundle_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "docs_dir": str(args.docs_dir),
        "files": files,
        "total_size_bytes": sum(f["size_bytes"] for f in files),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd()

    bundle_version = args.bundle_version
    # Late-resolve defaults that depend on bundle_version so users only
    # need to pass --bundle-version (not also --model-dir and --out).
    if args.model_dir is None:
        args.model_dir = f"dist/{bundle_version}/model"
    if args.out is None:
        args.out = f"dist/{bundle_version}/gtrends-bayes-{bundle_version}.tar.gz"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{bundle_version}-bundle-") as tmp:
        staging = Path(tmp)
        _copy_into_staging(
            repo_root, staging, Path(args.model_dir), Path(args.docs_dir),
            args.max_pickle_mb, bundle_version,
        )
        pkg_root = staging / f"gtrends-bayes-{bundle_version}"
        _verify_bundle_contents(pkg_root)

        manifest = _build_manifest(pkg_root, args, bundle_version)
        (pkg_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

        log.info(
            "staged %d files under %s (raw size %.2f MB)",
            len(manifest["files"]) + 1, pkg_root,
            manifest["total_size_bytes"] / 1024 / 1024,
        )

        # Tarball with max gzip compression.
        with tarfile.open(out_path, "w:gz", compresslevel=9) as tar:
            tar.add(pkg_root, arcname=pkg_root.name)

    compressed_mb = out_path.stat().st_size / 1024 / 1024
    log.info("wrote %s (%.2f MB compressed)", out_path, compressed_mb)
    if compressed_mb > args.max_mb:
        raise RuntimeError(
            f"bundle {compressed_mb:.2f} MB > {args.max_mb} MB cap. "
            "Strip more from the model pickles or trim supporting modules."
        )
    log.info("bundle OK (≤ %.1f MB email cap)", args.max_mb)
    log.info("SHA256: %s", _sha256(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
