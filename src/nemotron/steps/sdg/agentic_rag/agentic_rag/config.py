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

    bundle_column: str = "bundle"                 # evidence-set (gold chunks) for this row
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
    index_dir: Optional[str] = None               # cached embedding index; None = build in-memory
    corpus_path: Optional[str] = None             # chunk JSONL (for in-memory / lexical)
    top_k: int = 4                                # chunks returned per retrieval
    log_gold_rank: bool = True                    # record gold-chunk rank (grounding signal)
    guided_injection: bool = True                 # inject a missed gold chunk after a failed retry
    guided_injection_after: int = 2               # retries before injecting

    # ── depth & loop control (the "deep research" core) ───────────────────────
    min_hops: int = 2                             # floor on retrieval rounds (blocks lazy answers)
    max_steps: int = 6                            # hard ceiling on hops
    enforce_sufficiency: bool = True              # run the gap-check loop that drives depth up
    sufficiency_mode: Literal["strict", "soft"] = "strict"
    max_turns: int = 3                            # outer conversation follow-ups
    max_correction_attempts: int = 3             # tool-call schema-error retries

    # ── phased interaction ────────────────────────────────────────────────────
    interaction_model: Literal["strict_phased", "interleaved"] = "strict_phased"
    allow_discussion: bool = True                 # DISCUSSION phase (clarify before tool loop)
    max_discussion_exchanges: int = 3             # cap clarify turns before forcing the plan
    require_research_plan: bool = True            # assistant emits a plan at the transition

    # ── context management (sliding window + scratchpad) ──────────────────────
    context_window_k: int = 2                     # recent tool responses kept RAW in-view
    compaction_mode: Literal["reference", "summary", "drop"] = "reference"
    use_scratchpad: bool = True                   # carry distilled findings across hops
    store_full_trace: bool = True                 # training example keeps full chunks

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
            "user_query", "research_plan", "conversation_messages",
            "conversation_metadata", "conversation_status",
            "gold_rank_log", "hops_taken", "salvaged", "trajectory_judgment",
        ]

    @property
    def required_columns(self) -> List[str]:
        return [self.bundle_column, self.persona_column, self.theme_column, self.tools_column]
