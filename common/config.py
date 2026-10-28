"""YAML experiment configs and run provenance.

Each experiment is one YAML file in configs/. The runner copies the resolved config, the git commit,
the Python and package versions and the backends that actually ran into the results folder, so any
number in a results table can be traced back to exactly what produced it.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def git_info(cwd: str | Path = ".") -> dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "--verify", "-q", "HEAD")
    return {"commit": commit or None, "dirty": bool(run("status", "--porcelain"))}


def environment(packages: tuple[str, ...] = ("numpy", "scikit-learn", "sentence-transformers", "faiss-cpu",
                                              "pydantic", "pyyaml")) -> dict[str, Any]:
    versions = {}
    for p in packages:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions,
            "git": git_info(), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run_dir(root: str | Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(root) / name / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
