from __future__ import annotations

import argparse
from pathlib import Path

from plot_mix_metrics import PlotConfig, load_plot_config, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Mix samples where acc=1 and tool_use_num > iter_num."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("mix_plot_config.json")),
        help="Path to the graph config JSON.",
    )
    return parser.parse_args()


def print_matches(config: PlotConfig) -> None:
    records = load_records(config)
    matched = [
        record
        for record in records
        if record.acc >= 1.0 and record.tool_use_num > record.iter_num
    ]

    print(f"Matched samples: {len(matched)}")
    for record in matched:
        print(record.sample_uid)


def main() -> None:
    args = parse_args()
    config = load_plot_config(Path(args.config))
    print_matches(config)


if __name__ == "__main__":
    main()
