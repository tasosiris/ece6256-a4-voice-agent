"""Record the scripted utterances with your own voice, one WAV file per turn.

    python3 record.py --speaker alex --set a

For every line: read it, press Enter, say it, press Enter again. The file is 16 kHz mono 16-bit,
as the audio shim needs. The line you read is saved as the reference transcript in
recordings/references.json, grouped by speaker, which is what wer/score_wer.py reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sounddevice

from voice_kit.audio import DEFAULT_ENERGY_THRESHOLD, detect_speech_end_seconds, root_mean_square, write_wav

REPOSITORY = Path(__file__).resolve().parent
SAMPLE_RATE_HZ = 16_000

# The key press that starts and stops a take is cut off, so it is not taken for speech.
TRIM_START_SECONDS = 0.15
TRIM_END_SECONDS = 0.25


def record_until_enter(prompt: str = "") -> bytes:
    """Record from the default microphone until Enter is pressed; return 16-bit mono PCM."""
    chunks = []

    with sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE_HZ, channels=1, dtype="int16", callback=lambda data, *_: chunks.append(bytes(data))
    ):
        input(prompt)

    pcm = b"".join(chunks)
    first = int(TRIM_START_SECONDS * SAMPLE_RATE_HZ) * 2
    last = len(pcm) - int(TRIM_END_SECONDS * SAMPLE_RATE_HZ) * 2

    return pcm[first:last]


def take_problem(pcm: bytes) -> str | None:
    """Return why a take should be redone, or None when it is usable."""
    seconds = len(pcm) / (2 * SAMPLE_RATE_HZ)

    if seconds < 0.3:
        return "too short"

    try:
        speech_end = detect_speech_end_seconds(pcm)
    except ValueError:
        return "no speech above the shim's energy threshold: speak louder or move closer"

    # Speech "ending" at the very end means the room noise alone is above the threshold, and the
    # shim would then put the end of speech at the end of the file.
    quiet_tail = pcm[int(speech_end * SAMPLE_RATE_HZ) * 2 :]

    if seconds - speech_end < 0.05 and root_mean_square(pcm[-3200:]) >= DEFAULT_ENERGY_THRESHOLD:
        return "background noise is above the shim's threshold: find a quieter place or lower the input gain"

    print(f"    {seconds:.1f} s, speech ends at {speech_end:.1f} s, tail {len(quiet_tail) / 32:.0f} ms")

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, help="your name, lower case, no spaces")
    parser.add_argument("--set", required=True, help="which script set to read: a or b")
    parser.add_argument("--scripts", type=Path, default=REPOSITORY / "dialogues" / "scripts.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "recordings")
    parser.add_argument("--redo", help="record only this dialogue again, for example d03")
    arguments = parser.parse_args()

    dialogues = json.loads(arguments.scripts.read_text(encoding="utf-8"))[arguments.set]
    folder = arguments.output / arguments.speaker
    references_path = arguments.output / "references.json"
    references = json.loads(references_path.read_text(encoding="utf-8")) if references_path.exists() else {}
    mine = references.setdefault(arguments.speaker, {})

    print("Read each line naturally. Enter starts a take, Enter stops it. Stay quiet for a moment before stopping.")

    for dialogue in dialogues:
        if arguments.redo and dialogue["id"] != arguments.redo:
            continue

        print(f"\n== {dialogue['id']} ({dialogue['family']}) ==")

        for number, line in enumerate(dialogue["turns"], start=1):
            name = f"{arguments.speaker}_{dialogue['id']}_t{number:02d}.wav"

            while True:
                input(f'\n  Say: "{line}"\n  [Enter to start] ')
                pcm = record_until_enter("  recording... [Enter to stop] ")
                problem = take_problem(pcm)

                if problem:
                    print(f"    Redo: {problem}.")
                    continue

                if input("    [Enter to keep, r + Enter to redo] ").strip().lower() != "r":
                    break

            write_wav(folder / name, pcm, SAMPLE_RATE_HZ)
            sidecar = {"scenario_node": f"{dialogue['family']}.{dialogue['id']}"}
            (folder / name).with_suffix(".json").write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
            mine[name] = line
            references_path.write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved under {folder}; references in {references_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Takes already kept are saved.")
