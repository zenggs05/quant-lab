"""Build self-contained Kaggle notebooks from kaggle/submission.py.

Outputs (kaggle/build/):
  submission_model.py / .ipynb  - LightGBM model (MODE=model)
  submission_zeros.py / .ipynb  - zeros probe (MODE=zeros)
Usage: .venv/Scripts/python.exe -I kaggle/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_source(mode: str) -> str:
    src = (HERE / "submission.py").read_text(encoding="utf8")
    feats = (HERE / "features_pd.py").read_text(encoding="utf8")
    feats_body = feats.split('"""', 2)[2]  # drop module docstring
    feats_body = "\n".join(l for l in feats_body.splitlines() if not l.startswith(("import ", "from ")))
    w = json.loads((ROOT / "reports/eda_day2.json").read_text())["index_weights"]["weights_by_stock"]
    w_src = "INDEX_WEIGHTS = {" + ", ".join(f"{int(k)}: {v}" for k, v in sorted(w.items(), key=lambda x: int(x[0]))) + "}"
    a, rest = src.split("# FEATURES_PD_BEGIN\n")
    _, b = rest.split("# FEATURES_PD_END\n")
    src = a + feats_body.strip() + "\n\n" + b
    a, rest = src.split("# INDEX_WEIGHTS_BEGIN\n")
    _, b = rest.split("# INDEX_WEIGHTS_END\n")
    src = a + w_src + "\n" + b
    src = src.replace('os.environ.get("OPTIVER_MODE", "model")', f'os.environ.get("OPTIVER_MODE", "{mode}")')
    return src


def to_ipynb(code: str) -> dict:
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": code.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def main() -> None:
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    for mode in ("model", "zeros"):
        code = build_source(mode)
        compile(code, f"submission_{mode}.py", "exec")
        (out / f"submission_{mode}.py").write_text(code, encoding="utf8")
        (out / f"submission_{mode}.ipynb").write_text(json.dumps(to_ipynb(code), indent=1), encoding="utf8")
        print(f"wrote {out / f'submission_{mode}.ipynb'} ({len(code.splitlines())} lines)")


if __name__ == "__main__":
    main()
