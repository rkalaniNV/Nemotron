"""BFCL benchmark-family registration."""

from __future__ import annotations

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkFamilySpec
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.cli_orchestration import (
    run_bfcl_eval_cli,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
    generate_bfcl,
    prepare_bfcl,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.translation import (
    translate_bfcl,
)

SPEC = BenchmarkFamilySpec(
    name="bfcl",
    description="Function-calling benchmark generation from an executable oracle pack.",
    prepare_data=prepare_bfcl,
    generate=generate_bfcl,
    translate=translate_bfcl,
    evaluate=run_bfcl_eval_cli,
)
