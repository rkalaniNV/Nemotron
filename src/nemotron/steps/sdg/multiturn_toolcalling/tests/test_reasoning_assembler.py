"""Bounded reasoning gate + assembler / structured_messages projection."""

from __future__ import annotations

from mtsdg.assembler import assemble_blocks, project_structured_messages
from mtsdg.reasoning import validate_reasoning_content
from mtsdg.schemas import ClaimAndSupport, Message, ReasoningContent


def test_reasoning_grounding_rejects_fabricated_citation():
    # Citing a chunk that retrieve never returned is a HARD error (fabrication).
    rc = ReasoningContent(
        think="short trace",
        claims=[ClaimAndSupport(claim="x", supporting_chunk_ids=["c-not-returned"])],
    )
    v = validate_reasoning_content(rc, returned_chunk_ids={"c-auth-1"})
    assert not v.ok
    assert any("not returned" in e for e in v.errors)


def test_reasoning_uncited_claim_is_soft_warning():
    # A claim with no citation at all is a SOFT warning (does not reject).
    rc = ReasoningContent(think="trace", claims=[ClaimAndSupport(claim="x", supporting_chunk_ids=[])])
    v = validate_reasoning_content(rc, returned_chunk_ids={"c-auth-1"})
    assert v.ok
    assert any("no supporting_chunk_ids" in w for w in v.warnings)


def test_reasoning_accepts_grounded_within_budget():
    rc = ReasoningContent(
        think="grounded trace",
        claims=[ClaimAndSupport(claim="x", supporting_chunk_ids=["c-auth-1"])],
        answer_plan=["explain"],
    )
    v = validate_reasoning_content(rc, returned_chunk_ids={"c-auth-1"}, max_tokens=400)
    assert v.ok, v.errors


def test_reasoning_token_budget():
    rc = ReasoningContent(think="word " * 5000)
    v = validate_reasoning_content(rc, returned_chunk_ids=set(), max_tokens=50)
    assert not v.ok
    assert any("tokens" in e for e in v.errors)


def test_assemble_assigns_ids_and_projects():
    blocks = [[
        Message(role="user", content="hi", turn=1),
        Message(role="assistant", content="hello", turn=1, reasoning_content="think"),
    ]]
    system = Message(role="system", content="policy")
    assembled = assemble_blocks(blocks, system_message=system)
    assert assembled[0].message_id == "m-00"
    out = project_structured_messages(assembled)
    assert out[0]["role"] == "system"
    assert out[-1]["reasoning_content"] == "think"
    # Bookkeeping is stripped from the emitted messages.
    assert "turn" not in out[0] and "message_id" not in out[0]
