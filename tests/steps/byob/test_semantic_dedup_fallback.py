"""Shared Curator semantic-dedup fallback: Linux RAPIDS, CPU everywhere else."""

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemotron.steps.byob.runtime import deduplication as dedup_mod


class _Probe(dedup_mod.TextSemanticDeduplication):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.calls: list[str] = []

    def prepare_input_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset

    def _run_curator(self, input_file: str) -> None:
        del input_file
        self.calls.append("curator")

    def _run_cpu(self, dataset: pd.DataFrame) -> None:
        del dataset
        self.calls.append("cpu")

    def _mark_duplicates(self, dataset: pd.DataFrame) -> pd.DataFrame:
        copied = dataset.copy()
        copied["is_duplicate"] = False
        return copied


def _config(tmp_path: Any) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path),
        expt_name="probe",
        semantic_deduplication_config={
            "model_identifier": "unused",
            "n_clusters": 1,
            "eps": 0.07,
            "remove_duplicates": True,
        },
    )


def _dataset() -> pd.DataFrame:
    return pd.DataFrame({"id": [1], "text": ["hello"], "id_question": [1]})


def test_non_linux_hosts_use_the_cpu_backend(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dedup_mod.sys, "platform", "darwin")
    probe = _Probe(_config(tmp_path))

    result = probe.run(_dataset())

    assert probe.calls == ["cpu"]
    assert list(result["is_duplicate"]) == [False]


def test_linux_import_errors_fall_back_to_cpu(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dedup_mod.sys, "platform", "linux")

    class Missing(_Probe):
        def _run_curator(self, input_file: str) -> None:
            self.calls.append("curator")
            raise ImportError("rapids")

    probe = Missing(_config(tmp_path))

    probe.run(_dataset())

    assert probe.calls == ["curator", "cpu"]


def test_linux_value_errors_are_not_swallowed(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dedup_mod.sys, "platform", "linux")

    class Broken(_Probe):
        def _run_curator(self, input_file: str) -> None:
            raise ValueError("bad eps")

    probe = Broken(_config(tmp_path))

    with pytest.raises(ValueError, match="bad eps"):
        probe.run(_dataset())
