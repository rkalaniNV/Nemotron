"""DeepResearchSimulatorConfig — the single source of truth for every runtime knob.

Once the NDD pipeline starts, the generator reads *everything* from this object;
nothing is hard-coded in the loop. The outer YAML populates these fields, so
behaviour (depth, retrieval, context management, phases, quality gates) is fully
config-driven and reproducible.

The config extends Data Designer's ``SingleColumnConfig`` when DD is installed,
and falls back to a plain pydantic model otherwise so the pieces stay importable
and unit-testable without the DD runtime.
"""

from __future__ import annotations

from typing import List, Literal, Optional

try:  # real plugin path
    from data_designer.config.base import SingleColumnConfig as _Base
    _HAS_DD = True
except Exception:  # test / standalone path
    from pydantic import BaseModel as _Base  # type: ignore
    _HAS_DD = False

from pydantic import Field


class DeepResearchSimulatorConfig(_Base):
    """All knobs for the phased deep-research RAG simulation."""

    # ── column plumbing: where the generator reads seed/sampler data ──────────
    column_type: Literal["deep-research-simulator"] = "deep-research-simulator"
    model_alias: str = "assistant_model"          # DD requires a primary alias
    # direct OpenAI-client bypass (DD's facade mis-parses tool calls on some
    # endpoints): {alias: {model, base_url, api_key_env, params}} injected by step.py
    model_clients: dict = Field(default_factory=dict)

    bundle_column: str = "bundle"                 # evidence-set (gold chunks) — legacy seed path
    query_column: str = "seed_query"              # Stage-2 pre-generated query; "" -> persona invents
    cluster_id_column: str = "cluster_id"         # which cluster this row belongs to (index scope)
    gold_column: str = "gold_sections"            # JSON list of gold section ids (grounding)
    gold_doc_column: str = "gold_doc_ids"         # JSON list of gold document ids (generic grounding)
    query_level_column: str = "query_level"       # seed query difficulty kind (shapes opening turn)
    persona_column: str = "persona"               # Nemotron-Personas record
    theme_column: str = "theme"                   # conversation theme
    tools_column: str = "tools"                   # OpenAI-format tool schemas offered
    # diversity samplers (populated by DD category columns; read at runtime)
    archetype_column: str = "query_archetype"
    outcome_column: str = "outcome_type"
    depth_column: str = "depth_target"
    ambiguity_column: str = "ambiguity_level"

    # ── retrieval (RAG grounding) ─────────────────────────────────────────────
    retrieval_tools: List[str] = Field(default_factory=lambda: ["search_articles", "get_article"])
    retriever_backend: Literal["embedding", "lexical"] = "embedding"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_dir: Optional[str] = None               # single cached index (non-clustered runs)
    index_base_dir: Optional[str] = None          # per-cluster root: <base>/<cluster_id>/index
    corpus_path: Optional[str] = None             # chunk JSONL (for in-memory / lexical)
    top_k: int = 4                                # chunks returned per retrieval
    subsample_retrieval: bool = True              # Stage 3: oversample then random-subsample to top_k
    oversample_factor: int = 2                    # retrieve top_k*factor, keep a random top_k
    dedupe_retrieval: bool = True                 # exclude already-seen chunks so each hop brings new text
    log_gold_rank: bool = True                    # record gold-chunk rank (grounding signal)
    guided_injection: bool = False                # inject a missed gold chunk after failed retries
    guided_injection_after: int = 2               # retries before injecting

    # ── depth & loop control (the "deep research" core) ───────────────────────
    min_hops: int = 2                             # floor on retrieval rounds (blocks lazy answers)
    max_steps: int = 6                            # hard ceiling on hops
    enforce_sufficiency: bool = True              # run the gap-check loop that drives depth up
    sufficiency_mode: Literal["strict", "soft"] = "strict"
    force_first_tool: bool = True                 # force tool_choice=required until min_hops (grounding)
    max_turns: int = 3                            # legacy ceiling (superseded by conversation_plan)
    min_turns: int = 3                            # every conversation is at least this many turns
    max_correction_attempts: int = 3             # tool-call schema-error retries

    # ── conversation plan (Stage 4): sampled per row for length + type diversity ─
    # Declares num_turns range, the follow-up-kind distribution, and how each
    # query kind shapes its turn (depth/clarify). Empty => planner fallback.
    # { num_turns:{min,max}, follow_up_kinds:[{kind,weight}],
    #   kind_archetypes:{<kind>:{min_hops,max_hops,clarify,...}} }
    conversation_plan: dict = Field(default_factory=dict)

    # ── phased interaction ────────────────────────────────────────────────────
    interaction_model: Literal["strict_phased", "interleaved"] = "strict_phased"
    allow_discussion: bool = True                 # DISCUSSION phase (clarify before tool loop)
    max_discussion_exchanges: int = 3             # cap clarify turns before forcing the plan
    require_research_plan: bool = True            # assistant emits a plan at the transition

    # ── context management (Stage 5a: outer compression) ──────────────────────
    # The head (system prompt) and the tail (everything from the LAST user turn
    # to the end) are ALWAYS preserved verbatim; only the middle span is
    # compressed, and only once it exceeds the token budget below.
    context_window_k: int = 2                     # recent tool responses always kept RAW
    compaction_mode: Literal["reference", "summary", "drop"] = "reference"
    compression_token_limit: int = 2000           # budget (est. tokens) for the compressible middle
    preserve_last_user_turn: bool = True          # never compress from the last user turn onward
    use_scratchpad: bool = True                   # carry distilled findings across hops
    store_full_trace: bool = True                 # training example keeps full chunks
    parallel_tools: bool = True                   # run multiple tool calls in a turn concurrently
    max_tool_calls_per_turn: int = 1              # cap tool calls/turn (many NIM endpoints reject >1)

    # ── quality gates & yield ─────────────────────────────────────────────────
    gate_query: bool = True                       # judge/skip weak queries before the loop
    salvage_min_hops: int = 2                     # keep a late-failed trajectory past this depth
    majority_vote_n: int = 4                      # assistant self-consistency samples
    error_injection_rate: float = 0.0             # fraction of tool responses returning an error
    max_tools: int = 12                           # size of the tool subset offered per row

    # ── DD side-effect columns (outputs written back to the row) ──────────────
    @property
    def side_effect_columns(self) -> List[str]:
        return [
            "user_query", "research_plan", "conversation_messages", "conversation_messages_raw",
            "conversation_metadata", "conversation_status", "cluster_id",
            "gold_rank_log", "hops_taken", "salvaged", "trajectory_judgment",
            "conversation_plan",
        ]

    @property
    def required_columns(self) -> List[str]:
        # tools/persona/theme are always present (seed or sampler). The row's
        # question comes from either the Stage-2 query column or the legacy
        # bundle column, both read defensively at runtime.
        return [self.tools_column, self.persona_column, self.theme_column]
