"""Grounded glossary and cross-reference annotations after ingest."""

from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety.annotate import apply_annotations
from tests.steps.byob.safety_fixtures import scripted_complete_answer


class FakeCompleter:
    identity = {"provider": "fake", "model": "annotate"}

    def __init__(self, replies: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.requested: list[str] = []

    def complete(
        self,
        rows: list[Any],
        *,
        system_prompt: str = "",
        prompt: str = "",
        output_format: Any = None,
        prefix: str = "",
        inference_parameters: Any = None,
    ) -> dict[str, Any]:
        del system_prompt, prompt, inference_parameters
        if self.fail:
            raise ConnectionError("endpoint refused the connection")
        answers: dict[str, Any] = {}
        for row in rows:
            request_id = row["request_id"]
            self.requested.append(request_id)
            if request_id in self.replies:
                answers[request_id] = self.replies[request_id]
                continue
            structured = scripted_complete_answer(row, output_format=output_format, prefix=prefix)
            answers[request_id] = structured if structured is not None else {}
        return answers


class ExtractOnly:
    def extract(self, requests: list[Any]) -> dict[str, Any]:
        del requests
        return {}


def _span(text: str, *, span_id: str = "th#1#span-01", section: str = "1") -> dict[str, Any]:
    return {
        "source_span_id": span_id,
        "document_id": span_id.split("#", 1)[0],
        "section": section,
        "heading_path": [section],
        "kind": "paragraph",
        "text": text,
        "cross_references": [],
        "glossary_terms": [],
    }


def test_no_model_skips_annotation(tmp_path: Path) -> None:
    spans, terms, conflicts, report = apply_annotations([_span("Card: a number.")], [], [], None, tmp_path)

    assert report["status"] == "skipped"
    assert report["skipped_reason"] == "no_model"
    assert report["glossary_source"] == "ingest"
    assert terms == []
    assert spans[0]["glossary_terms"] == []


def test_an_extractor_without_complete_skips_annotation(tmp_path: Path) -> None:
    _spans, _terms, _conflicts, report = apply_annotations([_span("Card: a number.")], [], [], ExtractOnly(), tmp_path)

    assert report["skipped_reason"] == "extractor_cannot_complete"


def test_a_thai_colon_definition_is_kept_when_grounded(tmp_path: Path) -> None:
    span = _span("ข้อมูลบัตร: หมายเลขบัตรชำระเงิน")
    completer = FakeCompleter(
        {
            span["source_span_id"]: {
                "is_definition": True,
                "term_evidence": "ข้อมูลบัตร",
                "definition_evidence": "หมายเลขบัตรชำระเงิน",
                "cross_references": [],
            }
        }
    )

    spans, terms, conflicts, report = apply_annotations([span], [], [], completer, tmp_path)

    assert report["status"] == "applied"
    assert report["definitions"] == 1
    assert conflicts == []
    assert [(term["term"], term["definition"]) for term in terms] == [("ข้อมูลบัตร", "หมายเลขบัตรชำระเงิน")]
    assert spans[0]["glossary_terms"] == ["ข้อมูลบัตร"]


def test_ungrounded_annotation_is_dropped(tmp_path: Path) -> None:
    span = _span("ข้อมูลบัตร: หมายเลขบัตรชำระเงิน")
    completer = FakeCompleter(
        {
            span["source_span_id"]: {
                "is_definition": True,
                "term_evidence": "card data",
                "definition_evidence": "a payment number",
                "cross_references": [],
            }
        }
    )

    _spans, terms, _conflicts, report = apply_annotations([span], [], [], completer, tmp_path)

    assert report["definitions"] == 0
    assert terms == []


def test_a_grounded_citation_is_added_without_rewriting_span_ids(tmp_path: Path) -> None:
    clause = _span("Block requests for card data.", span_id="doc#4#span-01", section="4")
    referring = _span("See section 4 for the restriction.", span_id="doc#5#span-01", section="5")
    completer = FakeCompleter(
        {
            referring["source_span_id"]: {
                "is_definition": False,
                "term_evidence": "",
                "definition_evidence": "",
                "cross_references": [{"evidence": "See section 4", "target_section": "4"}],
            }
        }
    )

    spans, _terms, _conflicts, report = apply_annotations([clause, referring], [], [], completer, tmp_path)

    assert report["cross_references"] == 1
    assert [span["source_span_id"] for span in spans] == ["doc#4#span-01", "doc#5#span-01"]
    [reference] = spans[1]["cross_references"]
    assert reference["target_section"] == "4"
    assert reference["resolved"] is True
    assert reference["target_span_ids"] == ["doc#4#span-01"]


def test_prepare_glossary_is_not_rewritten(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"terms": [], "conflicts": []}', encoding="utf-8")
    span = _span("ข้อมูลบัตร: หมายเลขบัตรชำระเงิน")
    completer = FakeCompleter(
        {
            span["source_span_id"]: {
                "is_definition": True,
                "term_evidence": "ข้อมูลบัตร",
                "definition_evidence": "หมายเลขบัตรชำระเงิน",
                "cross_references": [],
            }
        }
    )

    apply_annotations([span], [], [], completer, tmp_path)

    assert glossary.read_text(encoding="utf-8") == '{"terms": [], "conflicts": []}'


def test_a_short_section_label_is_not_taken_from_a_longer_one(tmp_path: Path) -> None:
    clause = _span("Block requests for card data.", span_id="doc#4#span-01", section="4")
    nested = _span("Card data is restricted here.", span_id="doc#4.3#span-01", section="4.3")
    referring = _span("See section 4.3 for the restriction.", span_id="doc#5#span-01", section="5")
    completer = FakeCompleter(
        {
            referring["source_span_id"]: {
                "is_definition": False,
                "term_evidence": "",
                "definition_evidence": "",
                "cross_references": [
                    {"evidence": "See section 4.3", "target_section": "4"},
                    {"evidence": "See section 4.3", "target_section": "4.3"},
                ],
            }
        }
    )

    spans, _terms, _conflicts, report = apply_annotations([clause, nested, referring], [], [], completer, tmp_path)

    assert report["cross_references"] == 1
    [reference] = spans[2]["cross_references"]
    assert reference["target_section"] == "4.3"
    assert reference["resolved"] is True
    assert reference["target_span_ids"] == ["doc#4.3#span-01"]


def test_model_unavailable_keeps_ingest_terms(tmp_path: Path) -> None:
    ingest = [
        {
            "term": "restriction",
            "key": "restriction",
            "definition": "a limit",
            "document_id": "doc",
            "source_span_id": "doc#5#span-01",
        }
    ]
    clause = _span("Block requests for card data.", span_id="doc#4#span-01", section="4")
    referring = _span("See section 4 for the restriction.", span_id="doc#5#span-01", section="5")

    spans, terms, _conflicts, report = apply_annotations(
        [clause, referring], ingest, [], FakeCompleter(fail=True), tmp_path
    )

    assert report["status"] == "skipped"
    assert report["skipped_reason"] == "model_unavailable"
    assert report["glossary_source"] == "ingest"
    assert report["failures"] == []
    assert [(term["term"], term["definition"]) for term in terms] == [("restriction", "a limit")]
    assert spans[1]["glossary_terms"] == ["restriction"]
