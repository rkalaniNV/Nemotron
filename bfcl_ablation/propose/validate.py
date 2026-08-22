"""Gate a proposal before it can reach the pack.

Three gates, in the order a defect becomes cheaper to catch:

  schema     what the proposal says, judged against tools.json, fixtures.json and the
             pack's assertion list. Pure inspection, no execution.
  compile    what A1's rule table makes of it — `compile_milestones` either produces a
             milestone list or names the input it cannot infer from.
  plan       what `state_machine.build_plan` makes of that, which is the production
             check the oracle validator runs as check 1. Running it here is not a
             duplicate of the pipeline: it is the pipeline's own function, called early,
             because `generate_bfcl` refuses to run at all on a pack whose validation
             failed, so one bad proposal would otherwise cost the whole arm its numbers.

Nothing here repairs a proposal. A silently repaired proposal would be measured as an
acceptance, and the accept rate is the number this arm exists to report.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from bfcl_ablation.simplify import derive, rehydrate
from bfcl_ablation.simplify.milestones import compile_milestones

TEMPLATE_ID = re.compile(r"^[a-z][a-z0-9_]{3,60}$")
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
TEXT_MILESTONES = frozenset({"ask_for_slot", "ask_confirm", "decline", "final_answer"})
NO_TOOL_POLICIES = frozenset({"clarify_only", "irrelevant"})

REQUIRED_KEYS = (
    "template_id",
    "intent",
    "category",
    "difficulty",
    "turn_policy",
    "required_tools",
    "tools_present",
    "slots",
    "success_assertions",
    "user_turn_templates",
)
OPTIONAL_KEYS = (
    "call_order",
    "assistant_turn_templates",
    "corrects",
    "depends_on",
    "rationale",
)
# Carried through the prompt for the model's own use and dropped before the template is
# written; it is a proposal field, not a pack field.
NOT_PACK_FIELDS = ("rationale",)


class Rejected(ValueError):
    """A proposal the pack will not take, with the bucket the drop counts against."""

    def __init__(self, bucket: str, detail: str) -> None:
        super().__init__(detail)
        self.bucket = bucket
        self.detail = detail


def _reject(bucket: str, detail: str) -> None:
    raise Rejected(bucket, detail)


def _placeholders(text: str) -> set[str]:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.templating import placeholder_names

    return set(placeholder_names(text))


def _mentions_name(haystack: str, name: str) -> bool:
    """Whole-word tool-name match, agreeing with `render._mentions_name`.

    The production guard is the authority; this copy only has to reach the same verdict
    early enough that the proposal is dropped rather than costing the pack its gold.
    """
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", haystack) is not None


def _check_source(
    where: str,
    source: Any,
    *,
    fixtures: dict[str, list[dict[str, Any]]],
    tools: dict[str, dict[str, Any]],
) -> str:
    """Validate one slot source and return its kind."""
    if not isinstance(source, str) or not source.strip():
        _reject("schema_invalid", f"{where} declares no source")
    kind, _, rest = source.partition(":")
    if not rest:
        kind, rest = "fixture", source

    if kind == "fixture":
        collection, _, path_field = rest.partition(".")
        rows = fixtures.get(collection)
        if not isinstance(rows, list) or not rows:
            _reject("schema_invalid", f"{where} references unknown fixture collection {collection!r}")
        if not path_field or path_field not in rows[0]:
            _reject("schema_invalid", f"{where} references unknown field {path_field!r} of {collection!r}")
    elif kind == "absent":
        if rest.strip() not in fixtures:
            _reject("schema_invalid", f"{where} references unknown collection {rest.strip()!r}")
    elif kind == "literal":
        raw = rest.strip()
        if raw.startswith("["):
            try:
                values = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                _reject("schema_invalid", f"{where} declares an unparseable literal list")
            if not isinstance(values, list) or not values:
                _reject("schema_invalid", f"{where} declares an empty literal set")
        elif not raw:
            _reject("schema_invalid", f"{where} declares an empty literal")
    elif kind == "enum":
        tool_name, _, param = rest.partition(".")
        spec = tools.get(tool_name)
        if spec is None or not (spec["properties"].get(param) or {}).get("enum"):
            _reject("schema_invalid", f"{where} reads an enum {tool_name}.{param} that does not exist")
    elif kind == "range":
        if not rest.strip().startswith("{"):
            _reject("schema_invalid", f"{where} declares a malformed range")
    else:
        _reject("schema_invalid", f"{where} uses unsupported source kind {kind!r}")
    return kind


def validate_proposal(
    raw: Any,
    *,
    category: str,
    policy: str,
    tools: dict[str, dict[str, Any]],
    fixtures: dict[str, list[dict[str, Any]]],
    assertions: set[str],
    seen_ids: set[str],
) -> dict[str, Any]:
    """Return the authored template a proposal amounts to, or raise `Rejected`.

    `category` and `policy` are the sampled cell. A proposal that answers a different
    cell is refused rather than re-filed under the cell it chose: silently accepting it
    would let the model reshape the distribution the sampler exists to control, which is
    precisely the bias this arm is measuring.
    """
    if not isinstance(raw, dict):
        _reject("schema_invalid", f"proposal is {type(raw).__name__}, not an object")

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        _reject("schema_invalid", "missing keys: " + ", ".join(missing))
    unknown = sorted(set(raw) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        _reject("schema_invalid", "unknown keys: " + ", ".join(unknown))

    if str(raw["category"]) != category or str(raw["turn_policy"]) != policy:
        _reject(
            "cell_overridden",
            f"asked for ({category}, {policy}), answered with "
            f"({raw['category']}, {raw['turn_policy']})",
        )

    template_id = raw["template_id"]
    if not isinstance(template_id, str) or not TEMPLATE_ID.match(template_id):
        _reject("schema_invalid", f"template_id {template_id!r} is not a snake_case identifier")
    if template_id in seen_ids:
        _reject("duplicate_template_id", f"template_id {template_id!r} was already proposed")
    if not isinstance(raw["intent"], str) or not raw["intent"].strip():
        _reject("schema_invalid", "intent must be a non-empty string")
    if raw["difficulty"] not in DIFFICULTIES:
        _reject("schema_invalid", f"difficulty {raw['difficulty']!r} is not easy, medium or hard")

    required_tools = raw["required_tools"]
    tools_present = raw["tools_present"]
    for name, value in (("required_tools", required_tools), ("tools_present", tools_present)):
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            _reject("schema_invalid", f"{name} must be a list of tool names")
        unknown_tools = [item for item in value if item not in tools]
        if unknown_tools:
            _reject("schema_invalid", f"{name} names tools the pack does not have: {unknown_tools}")
    if not tools_present:
        _reject("schema_invalid", "tools_present is empty; the assistant must be offered some tool")
    outside = [name for name in required_tools if name not in tools_present]
    if outside:
        _reject("schema_invalid", f"required_tools {outside} are not in tools_present")
    if len(set(required_tools)) != len(required_tools):
        _reject("schema_invalid", "required_tools repeats a tool")

    success_assertions = raw["success_assertions"]
    if not isinstance(success_assertions, list) or not success_assertions:
        _reject("schema_invalid", "success_assertions must be a non-empty list")
    unknown_assertions = [name for name in success_assertions if name not in assertions]
    if unknown_assertions:
        _reject("schema_invalid", f"success_assertions names assertions the pack does not define: {unknown_assertions}")

    slots = raw["slots"]
    if not isinstance(slots, dict):
        _reject("schema_invalid", "slots must be an object")
    hidden: list[str] = []
    visible: list[str] = []
    for name, slot in slots.items():
        if not isinstance(slot, dict):
            _reject("schema_invalid", f"slot {name!r} must be an object")
        if not isinstance(slot.get("visible_in_first_turn"), bool):
            _reject("schema_invalid", f"slot {name!r} must declare visible_in_first_turn as a boolean")
        _check_source(f"slot {name!r}", slot.get("source"), fixtures=fixtures, tools=tools)
        if "filter" in slot and not isinstance(slot["filter"], str):
            _reject("schema_invalid", f"slot {name!r} filter must be a string expression")
        if "label" in slot and not (isinstance(slot["label"], dict) and slot["label"].get("vi")):
            _reject("schema_invalid", f"slot {name!r} label must carry Vietnamese text")
        extra = sorted(set(slot) - {"source", "visible_in_first_turn", "filter", "label"})
        if extra:
            _reject("schema_invalid", f"slot {name!r} declares unsupported keys: {extra}")
        (hidden if slot["visible_in_first_turn"] is False else visible).append(str(name))

    # A hidden slot is only ever collected by the missing_slot compiler. Under any other
    # policy the assistant would have to call with a value the conversation never states,
    # which the surface guards permit and no reader would accept.
    if hidden and policy != "missing_slot":
        _reject("schema_invalid", f"{policy} hides slots {sorted(hidden)}; only missing_slot may withhold a value")

    depends_on = raw.get("depends_on") or {}
    if not isinstance(depends_on, dict):
        _reject("schema_invalid", "depends_on must be an object")

    for tool_name in required_tools:
        for param in tools[tool_name]["required"]:
            if param not in slots and param not in depends_on:
                _reject(
                    "schema_invalid",
                    f"{tool_name} requires {param!r}, which no slot binds and no depends_on supplies",
                )

    turns = raw["user_turn_templates"]
    if not isinstance(turns, dict) or not str(turns.get("vi") or "").strip():
        _reject("schema_invalid", "user_turn_templates must carry non-empty Vietnamese text")
    text = str(turns["vi"])
    used = _placeholders(text)
    undeclared = sorted(used - set(slots))
    if undeclared:
        _reject("schema_invalid", f"user turn references undeclared slots {undeclared}")
    unstated = sorted(set(visible) - used)
    if unstated:
        _reject(
            "schema_invalid",
            f"user turn does not state visible slots {unstated}; the must_preserve guard drops it",
        )
    leaked = sorted(set(hidden) & used)
    if leaked:
        _reject("schema_invalid", f"user turn states withheld slots {leaked}")
    lowered = text.lower()
    named = sorted(name for name in tools if _mentions_name(lowered, name.lower()))
    if named:
        _reject("schema_invalid", f"user turn names the tools {named}, which the leakage guard forbids")

    assistant = raw.get("assistant_turn_templates") or {}
    if not isinstance(assistant, dict):
        _reject("schema_invalid", "assistant_turn_templates must be an object")
    for milestone, block in assistant.items():
        if milestone not in TEXT_MILESTONES:
            _reject("schema_invalid", f"assistant_turn_templates names unknown milestone {milestone!r}")
        if not isinstance(block, dict) or not str(block.get("vi") or "").strip():
            _reject("schema_invalid", f"assistant_turn_templates.{milestone} must carry Vietnamese text")
        stray = sorted(_placeholders(str(block["vi"])) - set(slots) - {"slot_name"})
        if stray:
            _reject("schema_invalid", f"assistant_turn_templates.{milestone} references undeclared slots {stray}")

    call_order = raw.get("call_order", "strict")
    if call_order not in {"strict", "any"}:
        _reject("schema_invalid", f"call_order {call_order!r} must be strict or any")
    if call_order == "any" and (policy == "dependent_call" or len(required_tools) < 2):
        _reject("schema_invalid", "call_order: any needs two independent calls")

    corrects = raw.get("corrects") or {}
    if not isinstance(corrects, dict):
        _reject("schema_invalid", "corrects must be an object")

    if policy in NO_TOOL_POLICIES:
        if required_tools:
            _reject("schema_invalid", f"{policy} declares required_tools; it must call nothing")
        if policy == "clarify_only" and not (assistant.get("ask_for_slot") or {}).get("vi"):
            _reject(
                "schema_invalid",
                "clarify_only must supply assistant_turn_templates.ask_for_slot; the pack-wide "
                "question names a withheld slot and this policy withholds none",
            )
    elif not required_tools:
        _reject("schema_invalid", f"{policy} declares no required_tools")

    if policy == "missing_slot" and not hidden:
        _reject("schema_invalid", "missing_slot withholds no slot")
    if policy == "multi_tool" and len(required_tools) < 2:
        _reject("schema_invalid", "multi_tool requires at least two tools")

    if policy == "correction":
        if not corrects:
            _reject("schema_invalid", "correction declares no `corrects`")
        for name, definition in corrects.items():
            if name not in slots:
                _reject("schema_invalid", f"corrects replaces undeclared slot {name!r}")
            body = definition if isinstance(definition, dict) else {"source": definition}
            kind = _check_source(f"corrects.{name}", body.get("source"), fixtures=fixtures, tools=tools)
            original = str(slots[name].get("source") or "")
            original_kind = original.partition(":")[0] if ":" in original else "fixture"
            if kind != original_kind:
                _reject(
                    "schema_invalid",
                    f"corrects.{name} resolves through {kind!r} but the slot resolves through "
                    f"{original_kind!r}; a replacement must resolve the same way",
                )
    elif corrects:
        _reject("schema_invalid", f"{policy} declares `corrects`, which only correction may do")

    if policy == "dependent_call" and not depends_on:
        _reject("schema_invalid", "dependent_call declares no `depends_on`")
    # Chaining is checked for coherence here, not for policy. `build_plan` accepts a
    # `from_result` marker under any policy, so refusing one at this gate would be this
    # module's opinion rather than the pipeline's contract.
    #
    # The run showed where the real rule lives: `expected_trace` rejects any template that
    # reads a prior result without declaring `dependent_call`, and eight proposals died
    # there rather than here. The gate is left permissive deliberately — the arm measures
    # the pipeline's contract, and a restatement of it that drifts would measure the
    # restatement — but the cost is that those eight are bucketed as `generation_failed`
    # instead of `schema_invalid`.
    if depends_on:
        if len(required_tools) < 2:
            _reject("schema_invalid", "depends_on needs a producer call and a consumer call")
        for param, spec in depends_on.items():
            if not isinstance(spec, dict) or not spec.get("from_call") or not spec.get("path"):
                _reject("schema_invalid", f"depends_on.{param} needs from_call and path")
            producer = str(spec["from_call"])
            if producer not in required_tools:
                _reject("schema_invalid", f"depends_on.{param} reads from {producer!r}, which is not required")
            if param in slots:
                _reject(
                    "schema_invalid",
                    f"depends_on.{param} also has a slot; a value cannot be both stated by the user "
                    "and read from a result",
                )
            consumers = [
                name
                for name in required_tools[required_tools.index(producer) + 1 :]
                if param in tools[name]["properties"]
            ]
            if not consumers:
                _reject("schema_invalid", f"no tool after {producer!r} accepts {param!r}")

    template = {key: raw[key] for key in REQUIRED_KEYS}
    for key in OPTIONAL_KEYS:
        if key in raw and key not in NOT_PACK_FIELDS and raw[key]:
            template[key] = raw[key]
    return template


def compile_and_plan(
    template: dict[str, Any],
    *,
    tools: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, str]],
    languages: list[str],
) -> dict[str, Any]:
    """Rehydrate the proposal through A1 and plan it through the production planner."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        PlanError,
        build_plan,
    )

    try:
        compile_milestones(template, tools)
        full = rehydrate.rehydrate_template(template, tools, canonical, languages)
    except derive.DerivationError as error:
        raise Rejected("milestone_compile_failed", str(error)) from error

    template_id = str(template["template_id"])
    try:
        build_plan(full, {"task_id": f"preflight:{template_id}", "template_id": template_id})
    except PlanError as error:
        raise Rejected("plan_invalid", str(error)) from error
    return full
