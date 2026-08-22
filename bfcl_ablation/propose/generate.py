"""Ask the model for task semantics, one sampled cell at a time.

The prompt hands over exactly what a human pack author would have in front of them —
the tool contracts, every fixture row, the assertion source, the ids that exist nowhere
— and then names the one thing the author does not get to choose: which (category,
policy) cell this proposal fills. Everything the model may vary is a semantic decision:
which tool answers the request, which record the slot binds, what the customer says,
which assertion pins the result.

The format example is deliberately from another domain. A banking example would hand
back the very choices the arm is trying to observe, and the tool/entity distribution
would then measure the example rather than the model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from bfcl_ablation.propose.sampler import POLICY_BRIEF, Cell

FORMAT_EXAMPLE = """\
{
  "template_id": "lib_hold_missing_branch",
  "intent": "place_hold_on_title",
  "category": "holds",
  "difficulty": "medium",
  "turn_policy": "missing_slot",
  "required_tools": ["place_hold"],
  "tools_present": ["place_hold", "get_title_status"],
  "slots": {
    "title_id": {"source": "fixture:titles.title_id", "visible_in_first_turn": true},
    "branch_id": {
      "source": "fixture:branches.branch_id",
      "visible_in_first_turn": false,
      "label": {"vi": "chi nhánh nhận sách"}
    }
  },
  "success_assertions": ["assert_hold_placed"],
  "user_turn_templates": {"vi": "Cho mình đặt giữ cuốn {title_id} nhé."},
  "rationale": "The branch is the value a real request forgets, so it is the one withheld."
}"""


def tool_catalogue(tools_raw: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in tools_raw:
        function = tool.get("function") or tool
        parameters = function.get("parameters") or {}
        flags = []
        if tool.get("x-mutates") or function.get("x-mutates"):
            flags.append("MUTATES STATE")
        if tool.get("x-requires-confirmation") or function.get("x-requires-confirmation"):
            flags.append("REQUIRES CONFIRMATION (pass confirm: true)")
        lines.append(
            f"- {function.get('name')}: {function.get('description', '')}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
        )
        for name, spec in (parameters.get("properties") or {}).items():
            need = "required" if name in (parameters.get("required") or []) else "optional"
            enum = spec.get("enum")
            lines.append(
                f"    {name} ({spec.get('type', 'any')}, {need})"
                + (f" one of {enum}" if enum else "")
            )
    return "\n".join(lines)


def fixture_catalogue(fixtures: dict[str, list[dict[str, Any]]], primary_keys: dict[str, str]) -> str:
    lines: list[str] = []
    for collection, rows in sorted(fixtures.items()):
        key = primary_keys.get(collection, "(no primary key)")
        lines.append(f"- {collection} (primary key: {key}, {len(rows)} rows)")
        for row in rows:
            lines.append("    " + json.dumps(row, ensure_ascii=False))
    return "\n".join(lines)


def edge_catalogue(edges: list[dict[str, str]], universe: tuple[str, ...]) -> str:
    inside = [e for e in edges if e["producer"] in universe and e["consumer"] in universe]
    if not inside:
        return "(no tool in this category returns a value another one requires)"
    return "\n".join(
        f"- {e['producer']} result path {e['path']!r} supplies {e['consumer']}.{e['parameter']}"
        for e in inside
    )


SYSTEM = """\
You author task templates for an executable Vietnamese banking benchmark. Each template
describes one customer request; a deterministic pipeline binds its slots against real
fixture rows, replays the resulting tool calls against a real backend, and keeps the
task only if the backend and the pack's assertions agree with the template. A template
that cannot survive that replay is worse than no template, so prefer a request the tools
can actually answer over an interesting one they cannot.

TOOLS
{tools}

FIXTURES (every row the backend holds)
{fixtures}

IDS THAT EXIST IN NO COLLECTION (write `absent:<collection>` to bind one)
{absent}

ASSERTIONS the pack already defines. You may only name assertions from this file, and
you must name one whose check matches what your task actually does. You do not write
assertions.
```python
{assertions}
```

TEMPLATE FIELDS
  template_id            unique snake_case id, 4-60 chars
  intent                 snake_case verb phrase, what the customer wants
  category, turn_policy  given to you; echo them back exactly
  difficulty             easy | medium | hard
  required_tools         tools the assistant MUST call, in call order
  tools_present          tools offered to the assistant; a superset of required_tools
  slots                  name -> {{source, visible_in_first_turn, filter?, label?}}
  success_assertions     one or more assertion names from the file above
  user_turn_templates    {{"vi": "..."}} the customer's opening message
  call_order             "strict" (default) or "any" for independent parallel calls
  assistant_turn_templates  overrides for the pack's wording, shaped
                         {{"<milestone>": {{"vi": "<the sentence the assistant says>"}}}}.
                         The only milestones are ask_for_slot, ask_confirm, decline and
                         final_answer; the value is always Vietnamese text, never a slot
                         name. Omit the field unless the policy notes ask for it.
  corrects               correction only: {{"<slot>": {{"source": "..."}}}}
  depends_on             dependent_call only: {{"<param>": {{"from_call": "<tool>",
                         "path": "<path into that tool's result>"}}}}
  rationale              one sentence on why this task is worth having

WHAT A SLOT IS
  A slot is a value the CUSTOMER supplies and the call passes. Every slot you declare is
  bound to a real value and handed to the tools. So:
    - do not declare a slot for an optional parameter you do not want passed; just leave
      it out, and the tool's own default applies;
    - `visible_in_first_turn: false` does not mean "optional" or "unimportant". It means
      the customer deliberately omits it from the opening message and the assistant has
      to ask. Only `missing_slot` may do that; under every other policy every slot must
      be visible_in_first_turn: true and must appear in the opening sentence.

SLOT SOURCES
  fixture:<collection>.<field>   bind a value from every matching row
  literal:[1000, 2000]           bind one of these constants (Python list syntax)
  absent:<collection>            bind an id that exists nowhere, for a not-found path
  enum:<tool>.<param>            bind one of a parameter's declared enum values
  A slot may add `filter: "balance_vnd >= 1000000"`, evaluated against the fixture row.

HARD RULES, each of which drops the template if broken
  1. Every required parameter of every required tool must be bound by a slot of the
     SAME NAME, or supplied by depends_on. There is no other way to fill an argument, so
     adding a tool to required_tools obliges you to bind every parameter it requires.
     Adding a tool you do not need is the most common way to lose a template.
  2. The opening message must literally contain {{slot}} for every slot with
     visible_in_first_turn: true, and must not contain a withheld one.
  3. The opening message must never name a tool.
  4. Only missing_slot may set visible_in_first_turn: false, and it withholds exactly the
     value the assistant then asks for.
  5. Bind ids through fixtures rather than typing them: a literal id the collection does
     not hold will fail the replay.
  6. Every assertion you name must hold for the trace your template actually produces.
     Naming an extra assertion about a call the template does not make will fail replay.
  7. Write natural Vietnamese. The customer never mentions functions, arguments or JSON.

Reply with JSON only: {{"proposals": [ ... ]}}, one object per requested proposal.

Format example, in an unrelated domain, for shape only:
```json
{example}
```"""

USER = """\
Cell to fill: category={category}, turn_policy={policy}
Proposals requested: {count}

Turn policy meaning: {brief}

Tools this category is about: {universe}
Result-to-argument links available inside this category:
{edges}

Write {count} proposal(s) for this cell and no other. Echo category={category} and
turn_policy={policy} exactly. {no_tool}{diversity}"""


def build_prompts(
    cell: Cell,
    *,
    tools_raw: list[dict[str, Any]],
    fixtures: dict[str, list[dict[str, Any]]],
    primary_keys: dict[str, str],
    absent_ids: dict[str, list[str]],
    assertions_source: str,
    edges: list[dict[str, str]],
) -> tuple[str, str]:
    system = SYSTEM.format(
        tools=tool_catalogue(tools_raw),
        fixtures=fixture_catalogue(fixtures, primary_keys),
        absent=json.dumps(absent_ids, ensure_ascii=False),
        assertions=assertions_source,
        example=FORMAT_EXAMPLE,
    )
    user = USER.format(
        category=cell.category,
        policy=cell.policy,
        count=cell.target,
        brief=POLICY_BRIEF[cell.policy],
        universe=", ".join(cell.universe) or "(none: this policy calls no tool)",
        edges=edge_catalogue(edges, cell.universe),
        no_tool=(
            "required_tools must be empty, but tools_present must still name two or three "
            "read-only tools the assistant was offered, so that not calling one is a "
            "judgement rather than an absence of options. "
            if cell.policy in {"clarify_only", "irrelevant"}
            else ""
        ),
        diversity=(
            "The proposals must differ from each other in the tool they call or the record "
            "they bind, not only in wording."
            if cell.target > 1
            else ""
        ),
    )
    return system, user


def _as_list(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("proposals", "templates", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    return list(payload) if isinstance(payload, list) else []


def propose(client: Any, cells: list[Cell], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Ask for every cell concurrently and return raw proposals tagged with their cell.

    A cell whose call fails outright yields no proposal rather than killing the batch;
    the shortfall is counted, because a cell the model would not answer is itself a
    coverage result.
    """

    def job(cell: Cell):
        system, user = build_prompts(cell, **context)

        def run() -> Any:
            return client.json_object(
                system=system,
                user=user,
                max_tokens=4000,
                temperature=0.0,
                seed=17,
            )

        return run

    outcomes = client.map([job(cell) for cell in cells])

    proposals: list[dict[str, Any]] = []
    for cell, payload in zip(cells, outcomes, strict=True):
        returned = _as_list(payload) if payload is not None else []
        for index in range(cell.target):
            proposals.append(
                {
                    "category": cell.category,
                    "policy": cell.policy,
                    "slot_index": index,
                    "raw": returned[index] if index < len(returned) else None,
                    "call_failed": payload is None,
                }
            )
    return proposals


COPIED_FROM_BASE = (
    "backend.py",
    "assertions.py",
    "tools.json",
    "fixtures.json",
    "manifest.yaml",
    "validation_cases.yaml",
)


def write_authored_pack(*, base: Path, target: Path, templates: list[dict[str, Any]]) -> Path:
    """Write an A1-shaped authored pack whose templates are the accepted proposals.

    Everything that is not a task — backend, tools, fixtures, assertions, the pack-wide
    wording — comes from the A1 pack unchanged. A3 varies one thing.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in COPIED_FROM_BASE:
        shutil.copy2(base / name, target / name)
    (target / "task_templates.yaml").write_text(
        yaml.safe_dump(templates, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target
