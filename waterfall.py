"""Check a trace, print the tables of the report and draw the latency waterfall.

    python3 waterfall.py --trace run_output/<job>/trace.jsonl --figure report/waterfall.png

Every number comes from the trace through voice_kit, the same code the TAs use. A turn with a
problem is listed and left out, as it is in the grading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_kit import (
    percentile,
    read_trace,
    record_problems,
    summarize_stages,
    summarize_time_to_first_audio,
)

ROWS = (
    ("time_to_first_audio", "**TTFA, all turns**"),
    ("asr", "ASR: `asr_final` minus `end_of_user_speech`"),
    ("llm_start", "LLM start: `llm_first_token` minus `asr_final`"),
    ("speech_start", "Speech start: `tts_first_audio` minus `llm_first_token`"),
    ("llm_remaining", "Not on the TTFA path: `llm_complete` minus `llm_first_token`"),
)

# What the agent's own extra stamps cut the two LLM-side stages into, as (name, later, earlier).
# A name without "_nanoseconds" is one of the five events.
PARTS = (
    ("action call (LLM picks the intent, not spoken)", "action_done_nanoseconds", "asr_final"),
    ("first token of the spoken reply", "llm_first_token", "action_done_nanoseconds"),
    ("first piece of text complete", "first_segment_nanoseconds", "llm_first_token"),
    ("Piper synthesis of the first piece", "tts_first_audio", "first_segment_nanoseconds"),
)

PATH_STAGES = (("asr", "ASR", "#2a78d6"), ("llm_start", "LLM start", "#eb6834"), ("speech_start", "Speech start", "#1baf7a"))
OFF_PATH_COLOR = "#c3c2b7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def stamp(record: dict, name: str) -> int | None:
    return record["extras"].get(name) if name.endswith("_nanoseconds") else record["events"].get(name)


def print_tables(records: list[dict], total: int) -> None:
    stages = summarize_stages(records)

    print("\n| | p50 (ms) | p95 (ms) |\n|---|---|---|")

    for name, title in ROWS:
        print(f"| {title} | {stages[name]['p50_milliseconds']:.0f} | {stages[name]['p95_milliseconds']:.0f} |")

    print(f"\nTurns counted: {len(records)} of {total}.")
    print(f"voice_kit.summarize_time_to_first_audio: {summarize_time_to_first_audio(records)}")

    print("\nInside the two LLM-side stages (from the agent's extra stamps):")
    print("| part | p50 (ms) | p95 (ms) |\n|---|---|---|")

    for title, later, earlier in PARTS:
        spans = [
            (stamp(record, later) - stamp(record, earlier)) / 1e6
            for record in records
            if stamp(record, later) is not None and stamp(record, earlier) is not None
        ]

        if spans:
            print(f"| {title} | {percentile(spans, 0.5):.0f} | {percentile(spans, 0.95):.0f} |")


def draw(records: list[dict], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = summarize_stages(records)
    figure, axes = plt.subplots(figsize=(8.2, 3.3), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    longest = 0.0

    for row, key in enumerate(("p50_milliseconds", "p95_milliseconds")):
        left = 0.0

        for name, title, color in PATH_STAGES:
            width = stages[name][key]
            # A thin surface-colored edge keeps touching segments apart without an outline.
            axes.barh(row, width, left=left, height=0.42, color=color, edgecolor=SURFACE, linewidth=2,
                      label=title if row == 0 else None)
            axes.text(left + width / 2, row - 0.3, f"{width:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
            left += width

        # The reply keeps streaming after the first token; that time is not on the path to first audio.
        start = stages["asr"][key] + stages["llm_start"][key]
        rest = stages["llm_remaining"][key]
        axes.barh(row + 0.36, rest, left=start, height=0.14, color=OFF_PATH_COLOR,
                  label="LLM remaining (not on the TTFA path)" if row == 0 else None)
        axes.text(start + rest + 8, row + 0.36, f"{rest:.0f}", va="center", fontsize=8, color=MUTED)

        ttfa = stages["time_to_first_audio"][key]
        axes.plot([ttfa, ttfa], [row - 0.27, row + 0.27], color=INK, linewidth=1.5,
                  label="TTFA measured over the turns" if row == 0 else None)
        axes.text(max(left, ttfa) + 0.02 * max(left, ttfa), row, f"TTFA {key[:3]} = {ttfa:.0f} ms", va="center", fontsize=10,
                  color=INK, fontweight="bold")
        longest = max(longest, left, ttfa, start + rest)

    axes.set_yticks([0, 1], ["p50", "p95"], color=INK)
    axes.set_ylim(1.75, -0.6)
    axes.set_xlim(0, longest * 1.32)
    axes.set_xlabel("milliseconds after the end of user speech", color=MUTED, fontsize=9)
    axes.set_title(
        f"Latency waterfall over {len(records)} turns: each segment is that stage's own percentile",
        loc="left", fontsize=10, color=INK,
    )
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, length=0, labelsize=9)

    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)

    axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=5, frameon=False, fontsize=8,
                labelcolor=MUTED, handlelength=1.2, columnspacing=1.2)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nfigure: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="where to write the waterfall figure (needs matplotlib)")
    arguments = parser.parse_args()

    records = read_trace(arguments.trace)
    valid = []

    for record in records:
        problems = record_problems(record)

        for problem in problems:
            print(record["turn_id"], problem)

        if not problems:
            valid.append(record)

    if not valid:
        sys.exit("No turn passes the checker.")

    print_tables(valid, len(records))

    if arguments.figure:
        draw(valid, arguments.figure)


if __name__ == "__main__":
    main()
