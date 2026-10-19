# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks for the Nemotron 3.5 Content Safety dataset factory."""

import json
from unittest.mock import Mock

import pytest

from nemotron.steps._runners.content_safety import (
    load_nemotron_content_safety_splits,
    nemotron_content_safety,
)


class _SafetyTokenizer:
    chat_template = "Native User Safety: classifier template"
    vocab_size = 1000
    eos_token = "<end_of_turn>"
    name_or_path = "test-native-tokenizer"
    init_kwargs = {"_commit_hash": "a" * 40}

    def get_vocab(self):
        return {"a": 1, "b": 2}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                          "request_categories": "/categories", "enable_thinking": False}
        return "<bos>classify:\n" + "\n".join(message["content"] for message in messages) + "\nmodel:\n"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())


def _safety_row(prompt="A neutral question", **changes):
    record = {"prompt": prompt, "response": None, "input_label": "safe", "response_label": None,
              "image_path": None, "violated_categories": None, "language": "en", "task_type": "safety",
              "row_id": prompt, "reasoning_trace": "Source-generated reasoning must not be a target"}
    record.update(changes)
    return record


def _safety_source():
    return {
        "train": [_safety_row(f"Training prompt {index}") for index in range(10)] + [
            _safety_row("Validation prompt 0", response="A different response", response_label="safe"),
            _safety_row("Training prompt 0", response="Duplicate prompt", response_label="safe"),
        ],
        "validation": [_safety_row(f"Validation prompt {index}") for index in range(4)],
        "test": [_safety_row(f"Test prompt {index}") for index in range(4)],
    }


class TestNemotronContentSafety:
    def test_supervises_only_labels_and_keeps_source_response_in_context(self):
        record = _safety_row(response="Source unsafe response", response_label="unsafe", violated_categories="Threat")
        actual = nemotron_content_safety(_SafetyTokenizer(), source_split="validation")(record)
        assert "Source unsafe response" in actual["input"]
        assert "Source unsafe response" not in actual["output"]
        assert record["reasoning_trace"] not in actual["input"] + actual["output"]
        assert actual["output"] == "User Safety: safe\nResponse Safety: unsafe\nSafety Categories: Threat<end_of_turn>\n"
        assert actual["metadata"]["source_split"] == "validation"
        assert actual["metadata"]["input_tokens"] == len(actual["input"].encode())
        assert actual["metadata"]["output_tokens"] == len(actual["output"].encode())

    def test_omits_absent_response_and_categories_when_all_safe(self):
        actual = nemotron_content_safety(_SafetyTokenizer())(_safety_row(violated_categories="Needs Caution"))
        assert actual["output"] == "User Safety: safe<end_of_turn>\n"

    @pytest.mark.parametrize("changes", [
        {"image_path": "image.jpg"}, {"task_type": "topic"}, {"language": "fr"},
        {"input_label": "harmful"}, {"response": "text"}, {"response_label": "safe"},
        {"violated_categories": "unknown category"}, {"prompt": ""},
    ])
    def test_excludes_rows_with_incompatible_modality_task_or_labels(self, changes):
        assert nemotron_content_safety(_SafetyTokenizer())(_safety_row(**changes)) is None

    def test_sampling_preserves_original_splits_and_reserves_all_heldout_prompts(self):
        source = _safety_source()
        kwargs = dict(dataset=source, train_size=4, validation_size=2, test_size=2)
        first = load_nemotron_content_safety_splits(_SafetyTokenizer(), **kwargs)
        second = load_nemotron_content_safety_splits(_SafetyTokenizer(), **kwargs)
        assert first == second
        seen = set()
        heldout = {record["prompt"] for split in ("validation", "test") for record in source[split]}
        for split, records in first.items():
            assert all(record["metadata"]["source_split"] == split for record in records)
            hashes = {record["metadata"]["prompt_sha256"] for record in records}
            assert len(hashes) == len(records)
            assert seen.isdisjoint(hashes)
            seen.update(hashes)
            if split == "train":
                assert all(record["messages"][0]["content"] not in heldout for record in records)

    def test_rejects_overlength_instead_of_truncating_the_classification_target(self):
        source = _safety_source()
        source["train"] = [_safety_row("x" * 1000)]
        with pytest.raises(ValueError, match="train: requested 1 valid safety rows, found 0"):
            load_nemotron_content_safety_splits(
                _SafetyTokenizer(), dataset=source, train_size=1, validation_size=1, test_size=1, seq_length=100,
            )

    def test_refuses_unpinned_source_or_incomplete_scan(self):
        with pytest.raises(ValueError, match="immutable commit"):
            load_nemotron_content_safety_splits(_SafetyTokenizer(), dataset=_safety_source(), revision="main")
        with pytest.raises(ValueError, match="refusing a partial scan"):
            load_nemotron_content_safety_splits(_SafetyTokenizer(), dataset=_safety_source(), max_scan_rows=2)

    def test_shared_cache_reuses_records_and_detects_corruption(self, tmp_path, monkeypatch):
        load = Mock(return_value=_safety_source())
        monkeypatch.setattr("datasets.load_dataset", load)
        kwargs = dict(cache_dir=tmp_path, train_size=2, validation_size=1, test_size=1)
        first = load_nemotron_content_safety_splits(_SafetyTokenizer(), **kwargs)
        assert load_nemotron_content_safety_splits(_SafetyTokenizer(), **kwargs) == first
        assert load.call_count == 1
        cache = next((tmp_path / "nemotron-content-safety").glob("*.json"))
        damaged = json.loads(cache.read_text())
        damaged["splits"]["train"][0]["output"] = "a corrupted classification target"
        cache.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="cache checksum/settings mismatch"):
            load_nemotron_content_safety_splits(_SafetyTokenizer(), **kwargs)

    def test_cache_identity_includes_sampling_and_native_template(self, tmp_path, monkeypatch):
        load = Mock(return_value=_safety_source())
        monkeypatch.setattr("datasets.load_dataset", load)
        tokenizer = _SafetyTokenizer()
        kwargs = dict(cache_dir=tmp_path, train_size=2, validation_size=1, test_size=1)
        load_nemotron_content_safety_splits(tokenizer, **kwargs)
        load_nemotron_content_safety_splits(tokenizer, seed=7, **kwargs)
        tokenizer.chat_template += " with a changed policy"
        load_nemotron_content_safety_splits(tokenizer, **kwargs)
        assert load.call_count == 3
        assert len(list((tmp_path / "nemotron-content-safety").glob("*.json"))) == 3

    def test_cache_identity_uses_actual_tokenizer_revision_and_vocabulary(self, tmp_path, monkeypatch):
        load = Mock(return_value=_safety_source())
        monkeypatch.setattr("datasets.load_dataset", load)
        tokenizer = _SafetyTokenizer()
        kwargs = dict(cache_dir=tmp_path, train_size=2, validation_size=1, test_size=1)
        load_nemotron_content_safety_splits(tokenizer, **kwargs)
        tokenizer.init_kwargs = {"_commit_hash": "b" * 40}
        load_nemotron_content_safety_splits(tokenizer, **kwargs)
        # Same size, EOS and template can still map text to different token IDs.
        tokenizer.get_vocab = lambda: {"a": 2, "b": 1}
        load_nemotron_content_safety_splits(tokenizer, **kwargs)
        assert load.call_count == 3
        parameters = [json.loads(path.read_text())["parameters"]
                      for path in (tmp_path / "nemotron-content-safety").glob("*.json")]
        assert {item["tokenizer"]["revision"] for item in parameters} == {"a" * 40, "b" * 40}
        assert len({item["tokenizer"]["backend_sha256"] for item in parameters}) == 2
