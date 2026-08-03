#!/usr/bin/env python3
"""Summarize Megatron-Bridge MODEL_TFLOP/s/GPU progress logs.

Also computes model FLOPs directly from a layer-resolved model shape, because
Megatron-Bridge mis-counts them for Nemotron-H hybrids: its FLOPs helper picks
layer counts from ``hybrid_layer_pattern``, but these models publish
``hybrid_override_pattern``, so it falls back to ``hybrid_attention_ratio`` /
``hybrid_mlp_ratio`` (both 0.0) and accounts for the model as all-Mamba with zero
MoE layers. The expert GEMMs -- the dominant term -- are dropped and the reported
MODEL_TFLOP/s (plus any MFU derived from it) collapses. Step time is measured by
an independent timer and stays trustworthy, so pairing a measured step time with
the shapes below recovers a usable throughput figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_THROUGHPUT_EVENTS = {"Saved checkpoint", "Saving async checkpoint"}


@dataclass(frozen=True)
class ProgressPoint:
    source: str
    timestamp: str
    event: str
    iteration: int
    world_size: int
    job_tflops_per_gpu: float
    cumulative_tflops_per_gpu: float
    floating_point_operations: float
    tokens_billions: float


@dataclass(frozen=True)
class ProgressSummary:
    source: str
    checkpoints: int
    latest_iteration: int
    world_size: int
    latest_job_tflops_per_gpu: float
    best_job_tflops_per_gpu: float
    latest_cumulative_tflops_per_gpu: float
    floating_point_operations: float
    tokens_billions: float
    elapsed_seconds: float
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    gpu_hours: float


def calculate_tflops_per_gpu(total_flops: float, elapsed_seconds: float, gpu_count: int) -> float:
    """Calculate aggregate work as TFLOP/s/GPU."""
    if total_flops < 0:
        raise ValueError("total_flops must be non-negative")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    return total_flops / elapsed_seconds / gpu_count / 1.0e12


@dataclass(frozen=True)
class HybridModelShape:
    """Layer-resolved shape of a hybrid Mamba / attention / MoE decoder.

    Layer counts are explicit rather than derived from a ratio, which is the whole
    point: the counts are what Megatron-Bridge gets wrong for these models.
    """

    name: str
    hidden_size: int
    padded_vocab_size: int
    num_attention_layers: int
    num_mamba_layers: int
    num_moe_layers: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    mamba_state_dim: int
    mamba_num_groups: int
    mamba_num_heads: int
    mamba_head_dim: int
    moe_ffn_hidden_size: int
    moe_router_topk: int
    num_shared_experts: int
    gated_linear_unit: bool = False

    @property
    def num_layers(self) -> int:
        return self.num_attention_layers + self.num_mamba_layers + self.num_moe_layers

    @property
    def shared_expert_ffn_hidden_size(self) -> int:
        return self.num_shared_experts * self.moe_ffn_hidden_size

    @property
    def mamba_d_inner(self) -> int:
        """Mamba inner width.

        Megatron-Bridge approximates this as ``2 * hidden_size``; Nemotron-H sizes
        it from the head geometry instead, which for Nano 3 is 4096 rather than
        5376, so the upstream heuristic over-counts every Mamba layer by ~31%.
        """
        return self.mamba_num_heads * self.mamba_head_dim


# Nemotron 3 Nano 30B-A3B Base: 52 layers as MEMEM*EMEMEM*... -> 23 Mamba, 23 MoE,
# 6 attention. Values from skills/nemotron-nano3 (paper Table 1); squared ReLU in
# the MoE blocks means the FFNs are ungated. Vocabulary is the padded 131072 the
# run reports, not the tokenizer's raw size.
NEMOTRON_3_NANO_30B_A3B = HybridModelShape(
    name="nemotron-3-nano-30b-a3b",
    hidden_size=2688,
    padded_vocab_size=131072,
    num_attention_layers=6,
    num_mamba_layers=23,
    num_moe_layers=23,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    mamba_state_dim=128,
    mamba_num_groups=8,
    mamba_num_heads=64,
    mamba_head_dim=64,
    moe_ffn_hidden_size=1856,
    moe_router_topk=6,
    num_shared_experts=2,
    gated_linear_unit=False,
)

MODEL_SHAPES: dict[str, HybridModelShape] = {
    NEMOTRON_3_NANO_30B_A3B.name: NEMOTRON_3_NANO_30B_A3B,
}

# Dense Tensor Core peaks (no structured sparsity) from NVIDIA product specs.
# Keep BF16 and FP8 separate: the same measured model FLOPs imply a different
# utilization percentage when the experiment changes compute precision.
HARDWARE_PEAK_TFLOPS: dict[str, float] = {
    "a100-80gb-bf16": 312.0,
    "h100-sxm-bf16": 989.4,
    "h100-sxm-fp8": 1978.9,
}

# Backward pass costs twice the forward pass. Activation recomputation adds real
# device work on top, but MODEL_TFLOPs deliberately excludes it so the metric stays
# comparable across recompute settings -- which is what the ablation needs.
_TRAINING_FLOPS_MULTIPLIER = 3.0


def _attention_flops_per_token(shape: HybridModelShape, seq_length: int) -> float:
    """Per-token FLOPs for one GQA layer, including the quadratic core term."""
    query_projection_size = shape.kv_channels * shape.num_attention_heads
    kv_projection_size = shape.kv_channels * shape.num_query_groups
    projections = 2.0 * (
        shape.hidden_size * (query_projection_size + 2 * kv_projection_size)
        + query_projection_size * shape.hidden_size
    )
    # QK^T and attn@V are 2 * 2 * qps * seq_length together, halved because causal
    # masking only evaluates the lower triangle. Matches Megatron's convention.
    core_attention = 2.0 * query_projection_size * seq_length
    return projections + core_attention


def _mamba_flops_per_token(shape: HybridModelShape) -> float:
    """Per-token FLOPs for one Mamba-2 layer (in_proj, scan, out_proj)."""
    d_inner = shape.mamba_d_inner
    in_proj_width = 2 * d_inner + 2 * shape.mamba_num_groups * shape.mamba_state_dim + shape.mamba_num_heads
    return (
        2.0 * shape.hidden_size * in_proj_width
        + 7.0 * d_inner * shape.mamba_state_dim
        + 2.0 * d_inner * shape.hidden_size
    )


def _moe_flops_per_token(shape: HybridModelShape) -> float:
    """Per-token FLOPs for one MoE layer: topk routed experts plus shared experts."""
    gate_multiplier = 1.5 if shape.gated_linear_unit else 1.0
    routed = 4.0 * shape.hidden_size * shape.moe_ffn_hidden_size * shape.moe_router_topk * gate_multiplier
    shared = 4.0 * shape.hidden_size * shape.shared_expert_ffn_hidden_size * gate_multiplier
    return routed + shared


def hybrid_flops_per_token(shape: HybridModelShape, seq_length: int) -> float:
    """Training FLOPs per token, summed over the real per-layer-type mix."""
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    forward = (
        shape.num_attention_layers * _attention_flops_per_token(shape, seq_length)
        + shape.num_mamba_layers * _mamba_flops_per_token(shape)
        + shape.num_moe_layers * _moe_flops_per_token(shape)
        # Output projection over the padded vocabulary.
        + 2.0 * shape.hidden_size * shape.padded_vocab_size
    )
    return forward * _TRAINING_FLOPS_MULTIPLIER


def hybrid_iteration_flops(shape: HybridModelShape, global_batch_size: int, seq_length: int) -> float:
    """Training FLOPs for one optimizer step at the given batch and sequence length."""
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    return hybrid_flops_per_token(shape, seq_length) * global_batch_size * seq_length


def calculate_runtime_metrics(
    floating_point_operations: float,
    cumulative_tflops_per_gpu: float,
    world_size: int,
    tokens_billions: float,
) -> dict[str, float]:
    """Derive elapsed time, token throughput, and GPU-hours from a progress record."""
    if cumulative_tflops_per_gpu <= 0:
        raise ValueError("cumulative_tflops_per_gpu must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if floating_point_operations < 0 or tokens_billions < 0:
        raise ValueError("work and token counts must be non-negative")

    elapsed_seconds = floating_point_operations / (
        cumulative_tflops_per_gpu * world_size * 1.0e12
    )
    if elapsed_seconds <= 0:
        raise ValueError("progress record must represent positive elapsed time")
    tokens_per_second = tokens_billions * 1.0e9 / elapsed_seconds
    return {
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_gpu": tokens_per_second / world_size,
        "gpu_hours": elapsed_seconds * world_size / 3600.0,
    }


def _field(parts: Sequence[str], name: str) -> str | None:
    prefix = f"{name}:"
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :].strip()
    return None


def _first_number(value: str | None) -> float:
    if value is None:
        raise ValueError("missing numeric field")
    return float(value.split()[0])


def parse_progress_line(line: str, source: str, line_number: int) -> ProgressPoint | None:
    """Parse one progress.txt line, ignoring events without throughput data."""
    parts = [part.strip() for part in line.strip().split("\t")]
    event = next((part for part in parts if part in _THROUGHPUT_EVENTS), None)
    if event is None:
        return None

    try:
        return ProgressPoint(
            source=source,
            timestamp=parts[0],
            event=event,
            iteration=int(_field(parts, "Iteration") or ""),
            world_size=int(_field(parts, "World size") or ""),
            job_tflops_per_gpu=_first_number(_field(parts, "Job throughput")),
            cumulative_tflops_per_gpu=_first_number(_field(parts, "Cumulative throughput")),
            floating_point_operations=_first_number(_field(parts, "Floating-point operations")),
            tokens_billions=_first_number(_field(parts, "Tokens (in billions)")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line_number}: malformed throughput record: {exc}") from exc


def parse_progress_file(path: Path) -> list[ProgressPoint]:
    points: list[ProgressPoint] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            point = parse_progress_line(line, str(path), line_number)
            if point is not None:
                points.append(point)
    return points


def summarize(points: Sequence[ProgressPoint]) -> ProgressSummary:
    if not points:
        raise ValueError("cannot summarize an empty progress log")
    latest = max(points, key=lambda point: point.iteration)
    runtime = calculate_runtime_metrics(
        latest.floating_point_operations,
        latest.cumulative_tflops_per_gpu,
        latest.world_size,
        latest.tokens_billions,
    )
    return ProgressSummary(
        source=latest.source,
        checkpoints=len(points),
        latest_iteration=latest.iteration,
        world_size=latest.world_size,
        latest_job_tflops_per_gpu=latest.job_tflops_per_gpu,
        best_job_tflops_per_gpu=max(point.job_tflops_per_gpu for point in points),
        latest_cumulative_tflops_per_gpu=latest.cumulative_tflops_per_gpu,
        floating_point_operations=latest.floating_point_operations,
        tokens_billions=latest.tokens_billions,
        **runtime,
    )


def discover_progress_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("progress.txt")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return list(dict.fromkeys(file.resolve() for file in files))


def _format_table(records: Sequence[dict[str, object]]) -> str:
    if not records:
        return ""
    columns = list(records[0])
    rows = [[str(record[column]) for column in columns] for record in records]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(columns), separator, *(render(row) for row in rows)])


def _write_records(records: Sequence[dict[str, object]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(records, indent=2))
        return
    if output_format == "csv":
        if not records:
            return
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        return
    print(_format_table(records))


def _augment_efficiency(
    record: dict[str, object],
    peak_tflops_per_gpu: float | None,
    target_tokens_billions: float | None,
) -> dict[str, object]:
    job_tflops = float(
        record.get("latest_job_tflops_per_gpu", record.get("job_tflops_per_gpu", 0.0))
    )
    if peak_tflops_per_gpu is not None:
        record["mfu_percent"] = 100.0 * job_tflops / peak_tflops_per_gpu

    if target_tokens_billions is not None:
        tokens_per_second = float(record["tokens_per_second"])
        world_size = int(record["world_size"])
        wall_hours = target_tokens_billions * 1.0e9 / tokens_per_second / 3600.0
        record["estimated_wall_hours"] = wall_hours
        record["estimated_gpu_hours"] = wall_hours * world_size
    return record


def _point_record(
    point: ProgressPoint,
    peak_tflops_per_gpu: float | None,
    target_tokens_billions: float | None,
) -> dict[str, object]:
    record = asdict(point)
    record.update(
        calculate_runtime_metrics(
            point.floating_point_operations,
            point.cumulative_tflops_per_gpu,
            point.world_size,
            point.tokens_billions,
        )
    )
    return _augment_efficiency(record, peak_tflops_per_gpu, target_tokens_billions)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read Megatron-Bridge progress.txt files and summarize native "
            "MODEL_TFLOP/s/GPU measurements."
        )
    )
    parser.add_argument("paths", nargs="*", help="progress.txt files or directories containing them")
    parser.add_argument(
        "--details",
        action="store_true",
        help="print every checkpoint record instead of one summary per progress file",
    )
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table")
    parser.add_argument("--total-flops", type=float, help="manual calculation: total floating-point operations")
    parser.add_argument("--elapsed-seconds", type=float, help="manual calculation: elapsed wall time")
    parser.add_argument("--gpus", type=int, help="manual calculation: number of GPUs")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SHAPES),
        help=(
            "model calculation: derive per-iteration FLOPs from a layer-resolved shape "
            "instead of trusting the pipeline's counter; requires --step-seconds, --gpus, "
            "--global-batch-size, and --seq-length"
        ),
    )
    parser.add_argument("--step-seconds", type=float, help="model calculation: measured seconds per iteration")
    parser.add_argument("--global-batch-size", type=int, help="model calculation: sequences per optimizer step")
    parser.add_argument("--seq-length", type=int, help="model calculation: tokens per sequence")
    parser.add_argument(
        "--peak-tflops-per-gpu",
        type=float,
        help="dense hardware peak used to report model FLOP utilization (MFU)",
    )
    parser.add_argument(
        "--hardware-peak",
        choices=sorted(HARDWARE_PEAK_TFLOPS),
        help=(
            "named dense Tensor Core peak used for MFU; use h100-sxm-bf16 for "
            "cases 21-25 and 28-30, or h100-sxm-fp8 for cases 26-27"
        ),
    )
    parser.add_argument(
        "--target-tokens-billions",
        type=float,
        help="target CPT token budget used to estimate wall time and GPU-hours",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.peak_tflops_per_gpu is not None and args.hardware_peak is not None:
        raise SystemExit("--peak-tflops-per-gpu and --hardware-peak are mutually exclusive")
    if args.hardware_peak is not None:
        args.peak_tflops_per_gpu = HARDWARE_PEAK_TFLOPS[args.hardware_peak]
    if args.peak_tflops_per_gpu is not None and args.peak_tflops_per_gpu <= 0:
        raise SystemExit("--peak-tflops-per-gpu must be positive")
    if args.target_tokens_billions is not None and args.target_tokens_billions <= 0:
        raise SystemExit("--target-tokens-billions must be positive")

    # `--gpus` is shared with the manual branch below, so this branch keys off
    # `--model` alone rather than on any of the model flags being present.
    model_only_values = (args.step_seconds, args.global_batch_size, args.seq_length)
    if args.model is not None:
        if args.gpus is None or any(value is None for value in model_only_values):
            raise SystemExit(
                "--model requires --step-seconds, --gpus, --global-batch-size, and --seq-length"
            )
        if args.total_flops is not None or args.elapsed_seconds is not None:
            raise SystemExit("--model cannot be combined with --total-flops or --elapsed-seconds")
        shape = MODEL_SHAPES[args.model]
        iteration_flops = hybrid_iteration_flops(shape, args.global_batch_size, args.seq_length)
        result = calculate_tflops_per_gpu(iteration_flops, args.step_seconds, args.gpus)
        tokens = args.global_batch_size * args.seq_length
        print(f"{shape.name}: {shape.num_layers} layers "
              f"({shape.num_mamba_layers} mamba, {shape.num_moe_layers} moe, "
              f"{shape.num_attention_layers} attention)")
        print(f"{iteration_flops:.4e} model FLOPs/iteration over {tokens} tokens")
        print(f"{result:.2f} MODEL_TFLOP/s/GPU")
        if args.peak_tflops_per_gpu is not None:
            print(f"{100.0 * result / args.peak_tflops_per_gpu:.2f}% MFU")
        return 0

    if any(value is not None for value in model_only_values):
        raise SystemExit("--step-seconds, --global-batch-size, and --seq-length require --model")

    manual_values = (args.total_flops, args.elapsed_seconds, args.gpus)
    if any(value is not None for value in manual_values):
        if not all(value is not None for value in manual_values):
            raise SystemExit("--total-flops, --elapsed-seconds, and --gpus must be provided together")
        if args.target_tokens_billions is not None:
            raise SystemExit("--target-tokens-billions requires progress.txt input")
        result = calculate_tflops_per_gpu(args.total_flops, args.elapsed_seconds, args.gpus)
        print(f"{result:.2f} MODEL_TFLOP/s/GPU")
        if args.peak_tflops_per_gpu is not None:
            print(f"{100.0 * result / args.peak_tflops_per_gpu:.2f}% MFU")
        return 0

    if not args.paths:
        raise SystemExit("provide at least one progress.txt file or directory")

    progress_files = discover_progress_files(args.paths)
    if not progress_files:
        raise SystemExit("no progress.txt files found")

    points_by_file = [(path, parse_progress_file(path)) for path in progress_files]
    empty_files = [str(path) for path, points in points_by_file if not points]
    if empty_files:
        print(
            "warning: no checkpoint throughput records in " + ", ".join(empty_files),
            file=sys.stderr,
        )

    if args.details:
        records = [
            _point_record(
                point,
                args.peak_tflops_per_gpu,
                args.target_tokens_billions,
            )
            for _, points in points_by_file
            for point in points
        ]
    else:
        records = [
            _augment_efficiency(
                asdict(summarize(points)),
                args.peak_tflops_per_gpu,
                args.target_tokens_billions,
            )
            for _, points in points_by_file
            if points
        ]
        records.sort(
            key=lambda record: float(record["tokens_per_second_per_gpu"]),
            reverse=True,
        )

    if not records:
        return 2
    _write_records(records, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
