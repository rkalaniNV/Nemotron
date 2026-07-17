"""Build the SFT file from the incremental checkpoint.

Use this to salvage completed trajectories if a run was killed/crashed before its
final write (the DD generator appends every finished episode to the checkpoint the
moment it completes).

Usage:
  python pipelines/from_checkpoint.py --checkpoint output/checkpoint.jsonl --out output/const_sft.jsonl
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="output/checkpoint.jsonl")
    ap.add_argument("--out", default="output/const_sft.jsonl")
    ap.add_argument("--include-rejected", action="store_true", help="also emit status=False trajectories")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.checkpoint, encoding="utf-8") if l.strip()]
    kept = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            if not args.include_rejected and r.get("trajectory_status") is not True:
                continue
            msgs = r.get("messages") or []
            if not msgs:
                continue
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            kept += 1
    accepted = sum(1 for r in records if r.get("trajectory_status") is True)
    print(f"checkpoint: {len(records)} episodes ({accepted} accepted) -> wrote {kept} -> {args.out}")
    for r in records:
        print(f"  {r.get('query_id')}: status={r.get('trajectory_status')} "
              f"n_messages={r.get('n_messages')} retrieved={r.get('n_retrieved_chunks')} "
              f"compactions={len(r.get('compaction_events') or [])}")


if __name__ == "__main__":
    main()
