"""Load environment and experiment configuration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def experiment_config() -> dict:
    path = REPO_ROOT / "config" / "experiment.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def neo4j_config() -> tuple[str, str, str]:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "dissertation2026")
    return uri, user, password
