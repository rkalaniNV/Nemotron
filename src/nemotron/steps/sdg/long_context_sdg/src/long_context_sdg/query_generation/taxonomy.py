"""Reviewed taxonomy loading and content hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from omegaconf import OmegaConf

from .schemas import QueryTaxonomy


def load_taxonomy(path: Path) -> tuple[QueryTaxonomy, str]:
    raw = path.read_bytes()
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    taxonomy = QueryTaxonomy.model_validate(payload)
    return taxonomy, hashlib.sha256(raw).hexdigest()
