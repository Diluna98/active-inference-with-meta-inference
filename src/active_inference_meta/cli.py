"""Command-line interface for the meta-inference example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import decisions_to_records, load_trace, run_meta_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run active meta-inference over a task observation trace."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="JSON trace; omit to use the packaged example.",
    )
    parser.add_argument("--initial-resolution", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    observations = load_trace(args.trace)
    decisions = run_meta_trace(
        observations,
        initial_resolution=args.initial_resolution,
    )
    records = decisions_to_records(observations, decisions)

    print("step  action  resolution  switched  confidence")
    for record in records:
        confidence = max(record["policy_posterior"])
        print(
            f"{record['step']:>4}  "
            f"{record['action_index']:>6}  "
            f"{record['selected_resolution']:>10}  "
            f"{record['switched']!s:>8}  "
            f"{confidence:>10.3f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"saved decisions: {args.output}")


if __name__ == "__main__":
    main()
