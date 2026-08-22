"""Compile assistant milestones from a template's intent fields.

`stages/state_machine._check_policy_shape` already encodes, per turn policy, the
structural constraint a milestone list must satisfy. This module is that validator
run backwards: given the policy and the tools the task must call, emit the minimal
milestone list the validator would accept.

Its input is (turn_policy, required_tools, call_order, slots, tool contract) — not
policy alone. Two policies additionally need input no schema implies, and those stay
authored:

  correction     which slot is corrected, and what it is corrected to
  dependent_call which argument is read from an earlier call's result, and at what
                 path into that result

Both are declared in one line each (`corrects:` / `depends_on:`) instead of a
hand-written milestone block, and both are flagged in the A1 report as the exact
places the rule table does not reach.
"""

from __future__ import annotations

from typing import Any

from bfcl_ablation.simplify.derive import DerivationError

# Policies whose conversation calls no tool. `build_plan` requires each to end in
# its own closing milestone, so the whole list is pinned by the policy name.
CLOSING_ONLY = {"irrelevant": "decline", "clarify_only": "ask_for_slot"}

# `_check_policy_shape` states a *constraint*, and a constraint is satisfied by many
# milestone lists, not one. So the compiler has to choose, and the validator does not
# say which. These are the choices, written down: a tie-breaking rule that lives only
# in control flow is a rule nobody can review.
#
# `verified_by` names the banking_vn templates whose hand-written milestones the rule
# reproduces exactly on the A1 round trip. A rule with an empty list is UNTESTED — it
# compiles, but no author has ever written the thing it claims to reproduce, so its
# output has never been checked against a human's intent.
RULE_TABLE = {
    "irrelevant": {
        "rule": "single `decline`; required_tools must be empty",
        "verified_by": ["bn_out_of_scope_service"],
    },
    "clarify_only": {
        "rule": "single `ask_for_slot`, which is also the closing milestone",
        "verified_by": ["bn_clarify_lookup_target"],
    },
    "single_turn": {
        "rule": "calls, then `final_answer`; no user turn after the opening one",
        "verified_by": ["bn_balance_single", "bn_card_limit_single", "bn_txn_status_single"],
    },
    "missing_slot": {
        "rule": (
            "one `ask_for_slot` per withheld slot, in slot declaration order, each "
            "answered by its own simulator turn, all before any call"
        ),
        "verified_by": ["bn_balance_withheld_account"],
        "untested": "the multi-slot branch: banking_vn withholds exactly one slot anywhere",
    },
    "confirmation": {
        "rule": "one `ask_confirm` covering the whole call batch, before any call",
        "verified_by": ["bn_create_transfer_single", "bn_create_dispute_single"],
        "untested": (
            "confirmation over a multi-call batch: every confirming template in "
            "banking_vn requires exactly one tool, so 'confirm once for the batch' "
            "versus 'confirm per call' has never been decided against a human's choice"
        ),
    },
    "correction": {
        "rule": (
            "`ask_confirm`(confirm_original), a simulator turn carrying slot_updates, "
            "`ask_confirm`(confirm_corrected), its reply, then the calls — the first "
            "approval is withdrawn by the correction so the call needs a second one"
        ),
        "verified_by": ["bn_transfer_amount_corrected"],
    },
    "multi_tool": {
        "rule": "call_order `any` puts every required tool in call_group 0; `strict` numbers them sequentially",
        "verified_by": ["bn_balance_and_card_parallel"],
    },
    "dependent_call": {
        "rule": (
            "sequential call groups in required_tools order; the consuming call takes "
            "its argument from `depends_on`, matched to the call that both precedes it "
            "and declares that parameter"
        ),
        "verified_by": ["bn_latest_txn_status_dependent"],
        "untested": "any milestone placed between the producing and consuming call",
    },
    "negative_path": {
        "rule": "same shape as the policy it shadows; the failure comes from the bound value, not the plan",
        "verified_by": ["bn_txn_status_unknown_id", "bn_transfer_short_of_funds"],
    },
}


def untested_rules() -> dict[str, str]:
    """Rules the round trip cannot vouch for, because the pack contains no such case."""
    return {policy: entry["untested"] for policy, entry in RULE_TABLE.items() if "untested" in entry}


def derive_args(
    tool_name: str,
    slots: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind the arguments `expected_trace` will not bind on its own.

    `expected_trace._bind_arguments` already fills any *required* parameter that has
    a same-named slot. What it deliberately does not do is reach for optional ones,
    so an optional parameter the template means to pass must be named explicitly —
    which is the entire content of the `args` blocks in the authored pack.

    The rule: pass an optional parameter iff a slot of that name exists, and pass
    `confirm: true` iff the tool declares it requires confirmation.
    """
    spec = tools.get(tool_name)
    if spec is None:
        raise DerivationError(f"template requires unknown tool {tool_name!r}")

    args: dict[str, Any] = {}
    for param in spec["properties"]:
        if param in spec["required"]:
            continue
        if param == "confirm" and spec["requires_confirmation"]:
            args["confirm"] = True
        elif param in slots:
            args[param] = "{" + param + "}"
    return args


def _withheld_slots(slots: dict[str, Any]) -> list[str]:
    return [name for name, slot in slots.items() if slot.get("visible_in_first_turn") is False]


def _call_milestones(
    template: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    required = [str(name) for name in (template.get("required_tools") or [])]
    call_order = str(template.get("call_order") or "strict")
    slots = template.get("slots") or {}
    depends_on = template.get("depends_on") or {}

    milestones: list[dict[str, Any]] = []
    consumed_dependencies: set[str] = set()
    for index, tool_name in enumerate(required):
        milestone: dict[str, Any] = {"type": "tool_call", "tool": tool_name}
        if call_order == "any":
            # One batch: `any` says the calls may be issued together, and a shared
            # group is what makes them a single parallel assistant turn.
            milestone["call_group"] = 0
        elif len(required) > 1:
            milestone["call_group"] = index
        args = derive_args(tool_name, slots, tools)

        for param, spec in depends_on.items():
            producer = str(spec.get("from_call") or "")
            if producer not in required:
                raise DerivationError(
                    f"template {template.get('template_id')!r} reads {param!r} from {producer!r}, "
                    "which is not one of its required_tools"
                )
            producer_index = required.index(producer)
            # The consuming call is the one that both takes this parameter and runs
            # after the call that produces it. Matching on the parameter name is what
            # keeps the producer from being handed its own output.
            if producer_index >= index or param not in (tools.get(tool_name) or {}).get("properties", {}):
                continue
            args[str(param)] = {
                "from_result": {"call": f"call_{producer_index}", "path": spec.get("path")}
            }
            consumed_dependencies.add(str(param))
        if args:
            milestone["args"] = args
        milestones.append(milestone)

    unconsumed = sorted(set(depends_on) - consumed_dependencies)
    if unconsumed:
        raise DerivationError(
            f"template {template.get('template_id')!r} declares depends_on for "
            f"{', '.join(unconsumed)}, which no later required tool accepts as a parameter"
        )

    # Only a call another call reads from needs a stable id.
    referenced = {f"call_{required.index(str(s.get('from_call')))}" for s in depends_on.values() if s.get("from_call") in required}
    for index, milestone in enumerate(milestones):
        if f"call_{index}" in referenced:
            milestone["id"] = f"call_{index}"
    return milestones


def compile_milestones(
    template: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (assistant_milestones, user_simulator_turns) for one template.

    Simulator replies carry no wording: the pack-wide canonical templates in the
    manifest supply it, and the compiler only records which slot each reply is about.
    """
    policy = str(template.get("turn_policy"))
    template_id = template.get("template_id")
    slots = template.get("slots") or {}

    closing = CLOSING_ONLY.get(policy)
    if closing is not None:
        if template.get("required_tools"):
            raise DerivationError(
                f"{policy} template {template_id!r} declares required_tools; a conversation that "
                "calls nothing cannot require a tool"
            )
        return [{"type": closing}], []

    required = [str(name) for name in (template.get("required_tools") or [])]
    if not required:
        raise DerivationError(
            f"{policy} template {template_id!r} declares no required_tools; only clarify_only and "
            "irrelevant may call nothing"
        )

    milestones: list[dict[str, Any]] = []
    simulator: list[dict[str, Any]] = []

    # 1. A withheld slot must be asked for, and answered, before any call.
    if policy == "missing_slot":
        withheld = _withheld_slots(slots)
        if not withheld:
            raise DerivationError(
                f"missing_slot template {template_id!r} withholds no slot; mark one "
                "visible_in_first_turn: false"
            )
        for name in withheld:
            milestone: dict[str, Any] = {"type": "ask_for_slot"}
            if len(withheld) > 1:
                milestone["slot"] = name
            milestones.append(milestone)
            simulator.append({"after": "ask_for_slot", "_reply": "provide_slot", "_slot": name})

    # 2. Confirmation. A correction withdraws the first approval, so it needs a second.
    needs_confirmation = any(tools.get(name, {}).get("requires_confirmation") for name in required)
    if policy == "correction":
        corrects = template.get("corrects") or {}
        if not corrects:
            raise DerivationError(
                f"correction template {template_id!r} declares no `corrects:`; which slot is "
                "replaced, and by what, is not implied by any schema"
            )
        milestones.append({"id": "confirm_original", "type": "ask_confirm"})
        slot_updates = {
            str(name): (spec if isinstance(spec, dict) else {"source": spec})
            for name, spec in corrects.items()
        }
        for name, spec in slot_updates.items():
            spec.setdefault("bind_as", f"{name}_corrected")
        simulator.append(
            {
                "after": "confirm_original",
                "_reply": "correct",
                "_slot": next(iter(slot_updates)),
                "slot_updates": slot_updates,
            }
        )
        milestones.append({"id": "confirm_corrected", "type": "ask_confirm"})
        simulator.append({"after": "confirm_corrected", "_reply": "confirm"})
    elif policy == "confirmation" or needs_confirmation:
        milestones.append({"type": "ask_confirm"})
        simulator.append({"after": "ask_confirm", "_reply": "confirm"})

    # 3. The calls themselves.
    milestones.extend(_call_milestones(template, tools))

    # 4. Every policy that calls a tool has to report on it.
    milestones.append({"type": "final_answer"})
    return milestones, simulator


def render_simulator_turns(
    simulator: list[dict[str, Any]],
    canonical: dict[str, dict[str, str]],
    languages: list[str],
) -> list[dict[str, Any]]:
    """Fill compiled simulator turns with the pack-wide canonical wording.

    `{slot}` in a canonical reply is replaced by a reference to the slot the reply is
    about, so one pack-wide sentence serves every template that asks for a value.
    """
    turns: list[dict[str, Any]] = []
    for entry in simulator:
        reply = entry["_reply"]
        pattern = canonical.get(reply)
        if pattern is None:
            raise DerivationError(f"manifest declares no canonical user reply for {reply!r}")
        slot = entry.get("_slot")
        content: dict[str, str] = {}
        for language in languages:
            text = str(pattern.get(language, ""))
            if not text:
                raise DerivationError(f"canonical reply {reply!r} has no {language!r} wording")
            if slot:
                alias = slot
                updates = entry.get("slot_updates") or {}
                if slot in updates:
                    alias = str(updates[slot].get("bind_as") or slot)
                text = text.replace("{slot}", "{" + alias + "}")
            content[language] = text
        turn: dict[str, Any] = {"after": entry["after"], "content_template": content}
        if entry.get("slot_updates"):
            turn["slot_updates"] = entry["slot_updates"]
        turns.append(turn)
    return turns
