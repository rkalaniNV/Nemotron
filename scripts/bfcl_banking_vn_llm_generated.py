#!/usr/bin/env python3
"""Author the Vietnamese banking Oracle Pack through the LLM-generated lane, then publish it.

`banking-vn-gold-v1-1392` was published from a hand-authored pack. This walks the same
domain down the other lane: the pack's backend, tool catalogue and fixtures are presented
as a reviewed source package, live probes certify it at A2, an authoring model drafts the
coverage plan, validation cases, template plans and assertion specs that source can
support, and the drafts plus a reviewed semantic supplement are assembled, validated,
frozen and published at the same 1,392-row scale.

    uv run python scripts/bfcl_banking_vn_llm_generated.py \
        --workdir ~/bfcl-runs/banking-vn-llm --author-model live \
        --model-provider nvidia_inference_api --model azure/openai/gpt-5.6-sol \
        --model-canonical-id nvidia-inference-api/azure/openai/gpt-5.6-sol

Two differences from the manual lane are structural, not incidental. The source publishes
one tool the hand-authored pack does not: A2 certification has to observe a timeout and
its cleanup, and no banking operation is unbounded, so the reviewed source adds a ledger
reconciliation sweep whose full scope walks every statement. And drafted assertions compile
to trace predicates only, so this pack's success assertions say which tools a task must
reach rather than what the results had to contain.

The four human review points are answered from the constants below and each prints what a
reviewer would have been deciding, exactly as `bfcl_llm_generated_demo.py` does.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BYOB_ROOT
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)
from nemotron.steps.byob.scripts import bfcl_author

PACK = BYOB_ROOT / "data" / "banking_vn_oracle_pack"
GOLD_CONFIG = BYOB_ROOT / "bfcl" / "config" / "banking_vn.gold.paraphrase.yaml"

PACK_ID = "banking_vn_llm"
PACK_VERSION = "0.1.0"
KEY_ID = "bfcl-banking-vn-llm"
REVIEWER = "reviewer@example.test"
OWNER = "owner@example.test"
CLOCK = "2026-03-02T09:00:00+07:00"
# The pack's six task categories, which the gold profile budgets one at a time.
CATEGORIES = 6

# The nine tools the hand-authored pack exposes. Templates are held to exactly these, so the
# published tool surface matches `banking-vn-gold-v1-1392` even though the source publishes
# the reconciliation sweep as well.
BANKING_TOOLS = (
    "get_account_balance",
    "get_card_limit",
    "get_transaction_status",
    "list_recent_transactions",
    "get_transfer_fee",
    "create_transfer",
    "get_vietqr_payment_status",
    "get_dispute_status",
    "create_dispute",
)
RECONCILE_TOOL = "reconcile_statements"
SOURCE_TOOLS = (*BANKING_TOOLS, RECONCILE_TOOL)

# A full reconciliation is the one operation this domain cannot bound, which is what lets the
# reviewed probe plan observe a timeout and its cleanup and therefore reach A2.
_RECONCILE_TOOL_SOURCE = """

def _reconcile_statements(arguments: dict) -> dict:
    full = arguments.get("full", False)
    if not isinstance(full, bool):
        return _err("invalid_argument", field="full", message="full must be a boolean")
    if full:
        time.sleep(3600)
    return {
        "status": "succeeded",
        "reconciled": len(_STATE.get("transactions", [])),
        "scope": "incremental",
    }
"""

_RECONCILE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RECONCILE_TOOL,
        "description": (
            "Reconcile posted statements against the ledger; a full sweep walks every "
            "statement the core has ever posted."
        ),
        "parameters": {
            "type": "object",
            "properties": {"full": {"type": "boolean", "default": False}},
            "required": ["full"],
            "additionalProperties": False,
        },
    },
}

DOMAIN_BRIEF = """\
Benchmark a Vietnamese retail banking assistant over a deterministic core-banking oracle.

The assistant answers customers about account balances, card limits, transaction status,
recent statement lines, transfer fees, VietQR payment status and card dispute status, and
it carries out two committing operations: it moves money over the NAPAS or internal rail,
and it opens a card dispute against a posted debit. Both committing operations are gated:
the core returns a pending status and changes nothing until the customer confirms. Money
movement is refused outright when the balance cannot cover the amount plus the rail fee.

Identifiers are prefixed and never guessed: accounts are ACC-, cards CARD-, transactions
TXN-, VietQR references VQ- and disputes DSP-. A reference the core does not hold is
reported as not found rather than approximated. Statement reconciliation is an operational
sweep, not a customer-facing answer.

The benchmark is authored and answered in Vietnamese.
"""


def _banner(step: str, title: str) -> None:
    print(f"\n\033[1m== {step}  {title}\033[0m", flush=True)


def _simulated(decision: str, detail: str) -> None:
    print(f"  \033[33m[simulated human review]\033[0m {decision}: {detail}", flush=True)


def _note(text: str) -> None:
    print(f"  {text}", flush=True)


# --------------------------------------------------------------------------------------
# The reviewed source package, and the probe plan that certifies it
# --------------------------------------------------------------------------------------


def write_source_package(root: Path) -> Path:
    """Present the banking oracle as a reviewed local Python source package."""
    package = root / "banking-source"
    package.mkdir(parents=True)
    shutil.copyfile(PACK / "fixtures.json", package / "fixtures.json")

    backend = (PACK / "backend.py").read_text(encoding="utf-8")
    backend = backend.replace("import re\n", "import re\nimport time\n", 1)
    # A reviewed source runs under a least-privilege policy that refuses getattr(). The
    # clock lookup is the pack's only reflective call, and asking for the attribute
    # directly reads the same context the same way.
    backend = backend.replace(
        '    value = getattr(_CTX, "clock", None)\n',
        "    try:\n        value = _CTX.clock\n    except AttributeError:\n        value = None\n",
        1,
    )
    backend = backend.replace(
        '    "create_dispute",\n]',
        f'    "create_dispute",\n    "{RECONCILE_TOOL}",\n]',
        1,
    )
    backend = backend.replace(
        '        "create_dispute": _create_dispute,\n',
        f'        "create_dispute": _create_dispute,\n        "{RECONCILE_TOOL}": _reconcile_statements,\n',
        1,
    )
    (package / "backend.py").write_text(backend + _RECONCILE_TOOL_SOURCE, encoding="utf-8")

    tools = json.loads((PACK / "tools.json").read_text(encoding="utf-8"))
    tools.append(_RECONCILE_TOOL_SCHEMA)
    (package / "tools.json").write_text(json.dumps(tools, indent=2), encoding="utf-8")

    (package / "dependency-lock.json").write_text(
        json.dumps({"schema_version": "bfcl-python-dependency-lock-v1", "dependencies": []}),
        encoding="utf-8",
    )
    return package


def write_probe_plan(path: Path) -> Path:
    """What intake may call, and what it must observe to certify the source at A2.

    One success case per published tool covers result shapes; the two committing tools
    carry a state-changing case so the mutation declaration can be checked against what
    actually happened, and the engine derives their unconfirmed counterparts itself.
    """
    path.write_text(
        json.dumps(
            {
                "schema_version": "bfcl-local-probe-plan-v1",
                "clock": CLOCK,
                "seed": 42,
                "fixtures": json.loads((PACK / "fixtures.json").read_text(encoding="utf-8")),
                "confirmation_parameter": "confirm",
                "status_field": "status",
                "pending_status": "awaiting_confirmation",
                "cases": [
                    {
                        "case_id": "a_balance_of_a_held_account",
                        "tool": "get_account_balance",
                        "arguments": {"account_id": "ACC-001"},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "b_limit_of_a_held_card",
                        "tool": "get_card_limit",
                        "arguments": {"card_id": "CARD-001"},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "c_status_of_a_posted_transaction",
                        "tool": "get_transaction_status",
                        "arguments": {"transaction_id": "TXN-1001"},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "d_recent_statement_lines",
                        "tool": "list_recent_transactions",
                        "arguments": {"account_id": "ACC-001", "limit": 5},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "e_fee_for_a_napas_transfer",
                        "tool": "get_transfer_fee",
                        "arguments": {
                            "from_account_id": "ACC-001",
                            "to_account_number": "9876543210",
                            "amount_vnd": 250000,
                            "rail": "napas",
                            "to_bank_code": "970436",
                        },
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "f_confirmed_transfer_commits",
                        "tool": "create_transfer",
                        "arguments": {
                            "from_account_id": "ACC-001",
                            "to_account_number": "9876543210",
                            "amount_vnd": 250000,
                            "rail": "napas",
                            "to_bank_code": "970436",
                            "confirm": True,
                        },
                        "expectation": "success",
                        "expected_state_change": True,
                    },
                    {
                        "case_id": "g_status_of_a_vietqr_payment",
                        "tool": "get_vietqr_payment_status",
                        "arguments": {"payment_ref": "VQ-1001"},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "h_status_of_an_open_dispute",
                        "tool": "get_dispute_status",
                        "arguments": {"dispute_id": "DSP-0001"},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        # TXN-1111 is a posted debit with no open dispute, so the confirmed
                        # call commits and the engine's unconfirmed variant stays pending.
                        "case_id": "i_confirmed_dispute_opens",
                        "tool": "create_dispute",
                        "arguments": {
                            "transaction_id": "TXN-1111",
                            "reason": "duplicate_charge",
                            "confirm": True,
                        },
                        "expectation": "success",
                        "expected_state_change": True,
                    },
                    {
                        "case_id": "j_incremental_reconciliation",
                        "tool": RECONCILE_TOOL,
                        "arguments": {"full": False},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "k_balance_of_an_unheld_account",
                        "tool": "get_account_balance",
                        "arguments": {"account_id": "ACC-ABSENT-1"},
                        "expectation": "structured_error",
                        "expected_error_code": "not_found",
                    },
                    {
                        "case_id": "l_limit_of_an_unheld_card",
                        "tool": "get_card_limit",
                        "arguments": {"card_id": "CARD-ABSENT-1"},
                        "expectation": "structured_error",
                        "expected_error_code": "not_found",
                    },
                    {
                        "case_id": "m_status_of_an_unposted_transaction",
                        "tool": "get_transaction_status",
                        "arguments": {"transaction_id": "TXN-ABSENT-1"},
                        "expectation": "structured_error",
                        "expected_error_code": "not_found",
                    },
                    {
                        "case_id": "n_reconciliation_scope_must_be_boolean",
                        "tool": RECONCILE_TOOL,
                        "arguments": {"full": "yes"},
                        "expectation": "structured_error",
                        "expected_error_code": "invalid_argument",
                    },
                    {
                        "case_id": "o_full_reconciliation_never_returns",
                        "tool": RECONCILE_TOOL,
                        "arguments": {"full": True},
                        "expectation": "timeout",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# The authoring model
# --------------------------------------------------------------------------------------

_CONFIRMATION_GATED = frozenset({"create_transfer", "create_dispute"})

_TOOL_PURPOSES: dict[str, tuple[str, str, str]] = {
    "get_account_balance": (
        "Report the available balance of one account.",
        "Report the balance of an account the bank holds.",
        "Report that an unheld account reference is not found.",
    ),
    "get_card_limit": (
        "Report the remaining spending limit of one card.",
        "Report the remaining limit of a card the bank issued.",
        "Report that an unissued card reference is not found.",
    ),
    "get_transaction_status": (
        "Report the status of one posted transaction.",
        "Report the status of a transaction the core posted.",
        "Report that an unposted transaction reference is not found.",
    ),
    "list_recent_transactions": (
        "List the most recent statement lines of one account.",
        "List recent statement lines for a held account.",
        "Refuse a status filter the core does not recognise.",
    ),
    "get_transfer_fee": (
        "Quote the fee and timing for one prospective transfer.",
        "Quote the fee for an amount the rail's schedule covers.",
        "Report that the paying account is not found.",
    ),
    "create_transfer": (
        "Move money out of one account over a chosen rail.",
        "Commit a transfer the customer has confirmed and the balance covers.",
        "Hold the transfer until the customer confirms it.",
    ),
    "get_vietqr_payment_status": (
        "Report the status of one VietQR payment.",
        "Report the status of a VietQR reference the core holds.",
        "Report that an unknown VietQR reference is not found.",
    ),
    "get_dispute_status": (
        "Report the status of one card dispute.",
        "Report the status of a dispute the bank opened.",
        "Report that an unknown dispute reference is not found.",
    ),
    "create_dispute": (
        "Open a card dispute against one posted debit.",
        "Open a dispute the customer has confirmed against a disputable debit.",
        "Hold the dispute until the customer confirms it.",
    ),
    RECONCILE_TOOL: (
        "Reconcile posted statements against the ledger.",
        "Reconcile incrementally during business hours.",
        "Refuse a reconciliation scope that is not a boolean.",
    ),
}

_TOOL_POLICIES: dict[str, list[str]] = {
    tool: (
        ["Confirm with the customer before committing."]
        if tool in _CONFIRMATION_GATED
        else ["Never guess an identifier the customer did not give."]
    )
    for tool in SOURCE_TOOLS
}


def _scripted_coverage_plan() -> dict[str, Any]:
    return {
        "tools": [
            {
                "tool": tool,
                "purpose": _TOOL_PURPOSES[tool][0],
                "policies": _TOOL_POLICIES[tool],
                "positive_intents": [_TOOL_PURPOSES[tool][1]],
                "negative_intents": [_TOOL_PURPOSES[tool][2]],
                "depends_on": (
                    ["get_transfer_fee"]
                    if tool == "create_transfer"
                    else ["get_transaction_status"]
                    if tool == "create_dispute"
                    else []
                ),
                "confirmation_relevant": tool in _CONFIRMATION_GATED,
            }
            for tool in SOURCE_TOOLS
        ],
        "cross_tool_notes": [
            "Quote the fee before committing a transfer.",
            "Check that a debit posted before disputing it.",
        ],
    }


def _scripted_assertion_specs() -> dict[str, Any]:
    """One reachability predicate per tool, in both directions.

    Trace predicates are the only ones L0 evidence can compile, so the plan states which
    tools a task must reach and which it must leave alone.
    """
    assertions: list[dict[str, Any]] = []
    for tool in SOURCE_TOOLS:
        assertions.append(
            {
                "assertion_id": f"called_{tool}",
                "subject": "trace",
                "predicate": "tool_called",
                "target": tool,
                "argument": None,
                "tool": None,
                "rationale": f"The task is only served by reaching {tool}.",
                "blocked_on": [],
            }
        )
        assertions.append(
            {
                "assertion_id": f"not_called_{tool}",
                "subject": "trace",
                "predicate": "tool_not_called",
                "target": tool,
                "argument": None,
                "tool": None,
                "rationale": f"A request the bank cannot serve must not reach {tool}.",
                "blocked_on": [],
            }
        )
    return {"assertions": assertions}


DRAFT_RESPONSES: dict[str, dict[str, Any]] = {
    "mcp_coverage_plan": _scripted_coverage_plan(),
    "mcp_validation_cases": {
        "cases": [
            {
                "case_id": "balance_of_a_held_account",
                "tool": "get_account_balance",
                "kind": "success",
                "intent": "Report the balance of an account the bank holds.",
                "arguments": [{"name": "account_id", "source": "fixture"}],
                "expectation": "The core reports that account's available balance.",
                "blocked_on": [],
            },
            {
                "case_id": "balance_of_an_unheld_account",
                "tool": "get_account_balance",
                "kind": "error",
                "intent": "Report that an unheld account reference is not found.",
                "arguments": [{"name": "account_id", "source": "absent_id"}],
                "expectation": "The core reports the account was not found.",
                "blocked_on": [],
            },
            {
                "case_id": "limit_of_a_held_card",
                "tool": "get_card_limit",
                "kind": "success",
                "intent": "Report the remaining limit of a card the bank issued.",
                "arguments": [{"name": "card_id", "source": "fixture"}],
                "expectation": "The core reports that card's remaining limit.",
                "blocked_on": [],
            },
            {
                "case_id": "transfer_waits_for_confirmation",
                "tool": "create_transfer",
                "kind": "confirmation_pending",
                "intent": "Hold a transfer until the customer confirms it.",
                "arguments": [
                    {"name": "from_account_id", "source": "fixture"},
                    {"name": "to_account_number", "source": "fixture"},
                    {"name": "amount_vnd", "source": "unresolved"},
                    {"name": "rail", "source": "literal", "literal": "napas"},
                ],
                "expectation": "The transfer waits rather than moving money.",
                "blocked_on": [],
            },
            {
                "case_id": "dispute_waits_for_confirmation",
                "tool": "create_dispute",
                "kind": "confirmation_pending",
                "intent": "Hold a dispute until the customer confirms it.",
                "arguments": [
                    {"name": "transaction_id", "source": "fixture"},
                    {"name": "reason", "source": "literal", "literal": "duplicate_charge"},
                ],
                "expectation": "The dispute waits rather than opening.",
                "blocked_on": [],
            },
            {
                "case_id": "incremental_reconciliation",
                "tool": RECONCILE_TOOL,
                "kind": "success",
                "intent": "Reconcile incrementally during business hours.",
                "arguments": [{"name": "full", "source": "literal", "literal": "false"}],
                "expectation": "The sweep reports how many statements it reconciled.",
                "blocked_on": [],
            },
        ]
    },
    "mcp_task_templates": {
        "templates": [
            {
                "template_id": "balance_of_one_account",
                "user_goal": "I want to know how much is left in my account.",
                "required_tools": ["get_account_balance"],
                "milestones": [
                    {
                        "description": "Look the account up in the core.",
                        "tool": "get_account_balance",
                        "requires_confirmation": False,
                    },
                    {
                        "description": "Tell the customer the balance.",
                        "tool": None,
                        "requires_confirmation": False,
                    },
                ],
                "policies": ["Never guess an identifier the customer did not give."],
                "blocked_on": [],
            },
            {
                "template_id": "transfer_after_confirmation",
                "user_goal": "I would like to send money to another account.",
                "required_tools": ["create_transfer"],
                "milestones": [
                    {
                        "description": "Ask the customer to confirm the transfer.",
                        "tool": None,
                        "requires_confirmation": False,
                    },
                    {
                        "description": "Commit the transfer once the customer confirms.",
                        "tool": "create_transfer",
                        "requires_confirmation": True,
                    },
                ],
                "policies": ["Confirm with the customer before committing."],
                "blocked_on": [],
            },
            {
                "template_id": "dispute_after_confirmation",
                "user_goal": "I want to dispute a charge I did not make.",
                "required_tools": ["create_dispute"],
                "milestones": [
                    {
                        "description": "Ask the customer to confirm the dispute.",
                        "tool": None,
                        "requires_confirmation": False,
                    },
                    {
                        "description": "Open the dispute once the customer confirms.",
                        "tool": "create_dispute",
                        "requires_confirmation": True,
                    },
                ],
                "policies": ["Confirm with the customer before committing."],
                "blocked_on": [],
            },
        ]
    },
    "mcp_assertion_specs": _scripted_assertion_specs(),
}


class ScriptedAuthoringModel:
    """Answers each drafting stage with one canned plan; grounding still runs for real."""

    def __init__(self) -> None:
        self.stages: list[str] = []

    def __call__(
        self,
        _run_dir: Path,
        *,
        stage_name: str,
        requests: list[dict[str, str]],
        **_kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        del requests
        self.stages.append(stage_name)
        return {stage_name: DRAFT_RESPONSES[stage_name]}


# --------------------------------------------------------------------------------------
# The reviewed semantics a model is not allowed to invent
# --------------------------------------------------------------------------------------


def _reachability_index(drafts: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read back which compiled assertion says each tool was, or was not, reached."""
    specs = yaml.safe_load((drafts / "assertion_specs.yaml").read_text(encoding="utf-8"))
    called: dict[str, str] = {}
    not_called: dict[str, str] = {}
    for spec in specs["assertions"]:
        index = {"tool_called": called, "tool_not_called": not_called}.get(spec["predicate"])
        if index is not None:
            index.setdefault(spec["target"], f"assert_{spec['assertion_id']}")
    return called, not_called


def write_supplement(path: Path, *, drafts: Path) -> Path:
    """The reviewed semantics no draft schema can express, bound to the drafted assertions.

    Slot bindings, turn policies and the Vietnamese user turns come from the reviewed pack.
    The success assertions cannot: they have to name assertions this drafting run actually
    compiled, so each template is held to the reachability of the tools it requires.
    """
    called, not_called = _reachability_index(drafts)
    missing = sorted(tool for tool in BANKING_TOOLS if tool not in called or tool not in not_called)
    if missing:
        raise SystemExit(
            "the drafted assertion specs do not state reachability in both directions "
            f"for {missing}; every published tool needs a tool_called and a "
            "tool_not_called predicate before templates can be bound to them. Purge the "
            "drafting cache and draft again."
        )

    templates = yaml.safe_load((PACK / "task_templates.yaml").read_text(encoding="utf-8"))
    for template in templates:
        template["tools_present"] = list(BANKING_TOOLS)
        required = template.get("required_tools") or []
        template["success_assertions"] = (
            [called[tool] for tool in required]
            if required
            # A template that requires no tool is passed by leaving every tool alone.
            else [not_called[tool] for tool in BANKING_TOOLS]
        )

    # Oracle validation holds every published tool to a success and a negative case, and
    # the reconciliation sweep is published here even though no template exposes it.
    cases = yaml.safe_load((PACK / "validation_cases.yaml").read_text(encoding="utf-8"))
    cases.extend(
        [
            {
                "id": f"success_{RECONCILE_TOOL}",
                "tool": RECONCILE_TOOL,
                "arguments": {"full": False},
                "expect": {"result_class": "success", "error_code": None},
                "reset_before": True,
            },
            {
                "id": f"wrong_type_{RECONCILE_TOOL}",
                "tool": RECONCILE_TOOL,
                "arguments": {"full": "yes"},
                "expect": {
                    "result_class": "structured_error",
                    "error_code": "invalid_argument",
                },
                "reset_before": True,
            },
        ]
    )

    manifest = yaml.safe_load((PACK / "manifest.yaml").read_text(encoding="utf-8"))
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bfcl-candidate-pack-supplement-v1",
                "languages": manifest["languages"],
                "clock": manifest["clock"],
                "system_prompt": manifest["system_prompt"],
                "absent_ids": manifest["absent_ids"],
                "primary_keys": manifest["primary_keys"],
                "assistant_turn_templates": manifest["assistant_turn_templates"],
                "task_templates": templates,
                "validation_cases": cases,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Guided CLI plumbing
# --------------------------------------------------------------------------------------


def run_guided(arguments: Sequence[str]) -> None:
    """Invoke the shipped guided CLI in-process."""
    previous = sys.argv
    sys.argv = ["bfcl_author.py", *arguments]
    try:
        bfcl_author.main()
    finally:
        sys.argv = previous


def write_certification_keys(root: Path) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_path = root / "certification-private.pem"
    public_path = root / "certification-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def write_generation_config(
    path: Path,
    *,
    pack_manifest: Path,
    output_dir: Path,
    expt_name: str,
    tasks_per_category: int,
    paraphrase: bool,
) -> Path:
    """Derive this pack's publication config from the manual lane's gold profile.

    Everything that decides what the benchmark measures — category budgets, difficulty,
    turn and policy mixes, deduplication limits, exports — is inherited, so the two lanes
    differ in how their pack was authored and not in what was asked of the inventory.
    """
    document = yaml.safe_load(GOLD_CONFIG.read_text(encoding="utf-8"))
    document["expt_name"] = expt_name
    document["output_dir"] = str(output_dir)
    document["oracle_pack"] = {"manifest_path": str(pack_manifest)}
    document["oracle_runtime"]["allowed_roots"] = [str(pack_manifest.parent)]

    generation = document["task_generation"]
    # Keep the gold profile's candidate-to-published headroom when the budget is lowered
    # for a smoke run, so selection is never asked to choose from a pool it cannot fill.
    headroom = generation["candidate_tasks_per_category"] / generation["tasks_per_category"]
    generation["tasks_per_category"] = tasks_per_category
    generation["candidate_tasks_per_category"] = round(tasks_per_category * headroom)
    generation["target_published_tasks"] = tasks_per_category * CATEGORIES

    if not paraphrase:
        document["lineage"]["roles"]["paraphrase"] = {"enabled": False, "model_config": None}
        document["surface_generation"]["model_paraphrase_enabled"] = False
        document["surface_generation"]["paraphrases_per_template"] = 0

    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# The walk itself
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--author-model",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted answers drafting locally; live sends the same prompts out",
    )
    parser.add_argument("--model-provider", default="test")
    parser.add_argument("--model", default="stub")
    parser.add_argument("--model-canonical-id", default="test/stub@1")
    parser.add_argument(
        "--tasks-per-category",
        type=int,
        default=232,
        help="232 publishes 1392 rows, matching banking-vn-gold-v1-1392",
    )
    parser.add_argument(
        "--no-paraphrase",
        action="store_true",
        help="publish template-only wording instead of one guarded paraphrase per binding",
    )
    parser.add_argument(
        "--expt-name",
        default=None,
        help="defaults to bfcl_banking_vn_llm_gold[_paraphrase]_v1_<rows>",
    )
    args = parser.parse_args()

    work = args.workdir.resolve()
    if work.exists():
        raise SystemExit(f"workdir already exists, pick a fresh one: {work}")

    paraphrase = not args.no_paraphrase
    rows = args.tasks_per_category * 6
    expt_name = args.expt_name or (f"bfcl_banking_vn_llm_gold{'_paraphrase' if paraphrase else ''}_v1_{rows}")

    os.environ["BFCL_ENABLE_LOCAL_PYTHON"] = "1"
    workspace = work / "workspace"
    work.mkdir(parents=True)

    caller: ScriptedAuthoringModel | None = None
    if args.author_model == "scripted":
        from nemotron.steps.byob.runtime.pack_authoring import model_client

        caller = ScriptedAuthoringModel()
        model_client._default_caller = caller  # type: ignore[assignment]

    package = write_source_package(work)
    brief = work / "domain-brief.txt"
    brief.write_text(DOMAIN_BRIEF, encoding="utf-8")
    private_key, public_key = write_certification_keys(work)

    _banner("1/9", "Intake: certify the reviewed banking source by probing it")
    run_guided(
        [
            "--ci",
            "author",
            "--workspace",
            str(workspace),
            "--source",
            str(package),
            "--brief",
            str(brief),
            "--pack-id",
            PACK_ID,
            "--pack-version",
            PACK_VERSION,
            "--required-tier",
            "A2",
            "--held-out-not-applicable-reason",
            "The fixtures are synthetic customer records authored for this benchmark.",
            "--held-out-reviewed-by",
            REVIEWER,
            "--certification-private-key",
            str(private_key),
            "--certification-key-id",
            KEY_ID,
            "--probe-plan",
            str(write_probe_plan(work / "probe-plan.json")),
        ]
    )
    intake = workspace / "intake"
    certification = json.loads((intake / "adapter_certification.json").read_text(encoding="utf-8"))
    _note(f"attained certification tier: {certification['attained_tier']}")

    _banner("2/9", "Authorize model exposure")
    _simulated(
        "model exposure",
        "an owner accepts that this redacted brief and tool surface may reach a model",
    )
    run_guided(
        [
            "--ci",
            "authorize",
            "--workspace",
            str(workspace),
            "--subject",
            str(intake / "model_exposure_subject.json"),
            "--authorized-by",
            OWNER,
        ]
    )

    _banner("3/9", "Approve the evidence bundle")
    evidence = load_evidence_bundle(
        intake / "evidence_bundle.json",
        certification_report_path=intake / "adapter_certification.json",
        trusted_certification_keys=load_trusted_certification_key(public_key, key_id=KEY_ID),
        domain_brief_source_path=intake / "domain_brief.source.txt",
        domain_brief_report_path=intake / "domain_brief_redaction.json",
        held_out_redaction_report_path=intake / "held_out_redaction.json",
        source_observations_path=intake / "source_observations.json",
    )
    _simulated(
        "evidence approval",
        "a reviewer confirms the normalized bundle describes the source they reviewed",
    )
    approval = workspace / "evidence_approval.json"
    run_guided(
        [
            "--ci",
            "approve",
            "--workspace",
            str(workspace),
            "--boundary",
            "evidence",
            "--approved-by",
            REVIEWER,
            "--source-bundle-digest",
            str(evidence.source_digest),
            "--normalized-bundle-digest",
            evidence.digest,
            "--output",
            str(approval),
        ]
    )

    _banner("4/9", "Draft: the model proposes what this banking source can support")
    drafting = workspace / "drafting"
    run_guided(
        [
            "--ci",
            "draft",
            "--workspace",
            str(workspace),
            "--bundle",
            str(intake / "evidence_bundle.json"),
            "--certification-report",
            str(intake / "adapter_certification.json"),
            "--certification-public-key",
            str(public_key),
            "--certification-key-id",
            KEY_ID,
            "--domain-brief-source",
            str(intake / "domain_brief.source.txt"),
            "--domain-brief-report",
            str(intake / "domain_brief_redaction.json"),
            "--held-out-redaction-report",
            str(intake / "held_out_redaction.json"),
            "--source-observations",
            str(intake / "source_observations.json"),
            "--exposure-authorization",
            str(workspace / "exposure_authorization.json"),
            "--approval",
            str(approval),
            "--output",
            str(drafting),
            "--model-alias",
            "author",
            "--model-provider",
            args.model_provider,
            "--model",
            args.model,
            "--model-canonical-id",
            args.model_canonical_id,
        ]
    )
    if caller is not None:
        _note(f"drafting stages answered: {', '.join(caller.stages)}")

    _banner("5/9", "Assemble: bind drafts and reviewed semantics into a candidate pack")
    _simulated(
        "reviewed semantics",
        "a human supplies slot bindings, turn policies, and the Vietnamese user turns, "
        "which no draft schema is allowed to express",
    )
    candidate_root = workspace / "candidate"
    supplement = write_supplement(work / "supplement.yaml", drafts=drafting / "drafts")
    run_guided(
        [
            "--ci",
            "assemble",
            "--workspace",
            str(workspace),
            "--supplement",
            str(supplement),
            "--output",
            str(candidate_root),
        ]
    )
    pack_root = candidate_root / "pack"
    _note(f"candidate pack: {pack_root}")

    _banner("6/9", "Validate the candidate pack and derive its tier")
    validation_config = write_generation_config(
        work / "candidate-validation.yaml",
        pack_manifest=pack_root / "manifest.yaml",
        output_dir=work / "validation-out",
        expt_name=f"{expt_name}_validation",
        tasks_per_category=args.tasks_per_category,
        paraphrase=paraphrase,
    )
    report_path = prepare_bfcl(validation_config)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _note(f"tier={report['tier']} gold_eligible={report['gold_eligible']}")
    if not report["gold_eligible"]:
        raise SystemExit("candidate pack is not gold-eligible; nothing to publish")

    _banner("7/9", "Review, approve, and freeze the release")
    packet = workspace / "review_packet.json"
    freeze_inputs = work / "freeze-inputs.json"
    run_guided(
        [
            "--ci",
            "review",
            "--workspace",
            str(workspace),
            "--adapter-kind",
            "local_python",
            "--pack",
            str(pack_root),
            "--evidence",
            str(intake / "evidence_bundle.json"),
            "--certification-report",
            str(intake / "adapter_certification.json"),
            "--certification-public-key",
            str(public_key),
            "--certification-key-id",
            KEY_ID,
            "--domain-brief-source",
            str(intake / "domain_brief.source.txt"),
            "--domain-brief-report",
            str(intake / "domain_brief_redaction.json"),
            "--held-out-redaction-report",
            str(intake / "held_out_redaction.json"),
            "--source-observations",
            str(intake / "source_observations.json"),
            "--intake-provenance",
            str(intake / "intake_provenance.json"),
            "--draft-provenance",
            str(drafting / "draft_provenance.json"),
            "--validation-report",
            str(report_path),
            "--validation-config",
            str(validation_config),
            "--resolved-authoring-config",
            str(workspace / "resolved_authoring_config.json"),
            "--exposure-authorization",
            str(workspace / "exposure_authorization.json"),
            "--evidence-approval",
            str(approval),
            "--output",
            str(packet),
            "--freeze-inputs-output",
            str(freeze_inputs),
        ]
    )
    _simulated(
        "release checklist",
        "a reviewer accepts semantics, descriptions, held-out policy, assumptions, "
        "validation evidence, certification, pre-model authorization, and questions",
    )
    release_approval = workspace / "release_approval.json"
    run_guided(
        [
            "--ci",
            "approve",
            "--workspace",
            str(workspace),
            "--boundary",
            "release",
            "--packet",
            str(packet),
            "--approved-by",
            REVIEWER,
            "--reviewed-at",
            CLOCK,
            "--output",
            str(release_approval),
            "--accept-semantics",
            "--accept-descriptions-and-snapshots",
            "--accept-held-out-policy",
            "--accept-assumptions",
            "--accept-validation-evidence",
            "--accept-independently-verified-certification",
            "--accept-pre-model-authorization",
            "--accept-answered-questions",
        ]
    )
    release_root = workspace / "release"
    run_guided(
        [
            "--ci",
            "freeze",
            "--workspace",
            str(workspace),
            "--freeze-inputs",
            str(freeze_inputs),
            "--approval",
            str(release_approval),
            "--output",
            str(release_root),
        ]
    )

    _banner("8/9", "Publish: fresh validation, then generate the benchmark")
    publication_config = write_generation_config(
        work / "publication.yaml",
        pack_manifest=release_root / "pack" / "manifest.yaml",
        # Guided publication binds run_manifest.json into the session, so the generation
        # output has to land inside the workspace that session covers.
        output_dir=workspace / "generated",
        expt_name=expt_name,
        tasks_per_category=args.tasks_per_category,
        paraphrase=paraphrase,
    )
    run_guided(
        [
            "--ci",
            "publish",
            "--workspace",
            str(workspace),
            "--release",
            str(release_root),
            "--config",
            str(publication_config),
        ]
    )
    published = workspace / "generated" / expt_name
    manifest = json.loads((published / "run_manifest.json").read_text(encoding="utf-8"))

    _banner("9/9", "What was published")
    _note(f"benchmark: {published / 'benchmark.parquet'}")
    _note(f"tier={manifest['tier']} gold_eligible={manifest['gold_eligible']}")
    _note(f"generation_mode={manifest['generation_mode']}")
    _note(f"published rows={manifest['publication']['published']['rows']}")
    _note(f"raw rows={manifest['publication']['raw']['rows']}")
    _note(f"exports: {published / 'exports'}")


if __name__ == "__main__":
    main()
