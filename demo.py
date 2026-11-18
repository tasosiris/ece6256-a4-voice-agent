"""Talk to the agent live with a microphone and speakers, for the demo video (not for measuring).

    python3 demo.py --llm-model llama3.1:8b --llm-base-url http://127.0.0.1:11434/v1

Push-to-talk with the Enter key: Enter to start talking, Enter when done. Type n + Enter to start a
new conversation and q + Enter to quit. Latency numbers come from the audio shim (agent.py), never
from this script.

Before every turn the script shows a line to say, chosen from what the agent just asked, and walks
through one timer, one lookup and one booking. --no-guide hides this.
"""

from __future__ import annotations

import argparse
import time

import sounddevice

from agent import VoiceAgent, add_agent_arguments
from record import record_until_enter

# The three tasks of the demo: a name, the opening line, then (words in the agent's reply, what to
# say next). None as the next line means the task is complete. The first matching pair is used.
GUIDE = (
    ("a timer", "Set a timer for five minutes for the eggs.", (
        ("starting now", "How long is left?"),
        ("How long should", "Five minutes."),
        ("left", None),
    )),
    ("a lookup question", "What time does the cafe close on Saturdays?", (
        ("closes", None),
    )),
    ("a room booking", "I need a room on Thursday afternoon.", (
        ("Booked.", None),
        ("How many people", "Five."),
        ("equipment", "A projector."),
        ("Shall I book it", "Yes, under the name Sam."),
        ("what name", "Sam."),
        ("What time", "Two in the afternoon."),
        ("Which day", "Thursday afternoon."),
    )),
)

HELP = """
How this works
  1. Press Enter, say the line shown after SAY, wait a moment, press Enter again.
  2. The agent answers out loud. If it asks you something, the next SAY line is the answer.
  3. You do not have to give every detail at once: the agent asks for what is missing, one
     question at a time, and remembers what you already said.
  4. If it mishears you, answer its question in your own words, or type n + Enter to start the
     task again.
"""


def next_line(task: int, reply: str) -> tuple[str | None, bool]:
    """Return (what to say next, whether the task is complete) after the agent's reply."""
    for words, line in GUIDE[task][2]:
        if words.lower() in reply.lower():
            return line, line is None

    return None, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_agent_arguments(parser)
    parser.add_argument("--no-guide", action="store_true", help="do not show what to say")
    arguments = parser.parse_args()
    agent = VoiceAgent(arguments)
    agent.warm_up()

    # The task being shown and the line to say next; task is None when no guide is wanted or left.
    task = None if arguments.no_guide else 0
    line = None if task is None else GUIDE[0][1]

    if task is not None:
        print(HELP)
        print(f"Task 1 of {len(GUIDE)}: {GUIDE[0][0]}.")

    with sounddevice.RawOutputStream(samplerate=agent.sample_rate_hz, channels=1, dtype="int16") as speaker:
        while True:
            if line:
                print(f"\nSAY:  {line}")

            command = input("[Enter to talk, n = new conversation, q = quit] ").strip().lower()

            if command == "q":
                break

            if command == "n":
                agent.reset()
                print("New conversation.")
                line = None if task is None else GUIDE[task][1]
                continue

            pcm = record_until_enter("listening... [Enter when done] ")
            released = time.monotonic()
            first_audio = []

            def play(chunk: bytes) -> None:
                first_audio.append(time.monotonic())
                speaker.write(chunk)

            extras = agent.respond(pcm, lambda event: None, play)
            delay = f"{(first_audio[0] - released) * 1000:.0f} ms to first audio" if first_audio else "no audio"
            print(f"you:   {extras['transcript']}\nagent: {extras['reply']}\n({delay})")

            if task is None:
                continue

            line, complete = next_line(task, extras["reply"])

            if complete:
                # Each task is its own conversation, as in the recorded dialogues.
                agent.reset()
                task = task + 1 if task + 1 < len(GUIDE) else None

                if task is None:
                    print("\nAll three tasks shown. Type q + Enter to quit.")
                else:
                    line = GUIDE[task][1]
                    print(f"\nDone. Task {task + 1} of {len(GUIDE)}: {GUIDE[task][0]}.")

            elif line is None:
                print("\nNo suggestion for that reply: answer the agent in your own words, or type n + Enter to redo the task.")


if __name__ == "__main__":
    main()
