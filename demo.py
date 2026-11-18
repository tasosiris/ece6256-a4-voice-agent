"""Talk to the agent live with a microphone and speakers, for the demo video (not for measuring).

    python3 demo.py --llm-model llama3.1:8b --llm-base-url http://127.0.0.1:11434/v1

Push-to-talk with the Enter key: Enter to start talking, Enter when done. Type n + Enter to start a
new conversation and q + Enter to quit. Latency numbers come from the audio shim (agent.py), never
from this script.

Before every turn the script shows a line to say, chosen from what the agent just asked, and walks
through one timer, one lookup and one booking. --no-guide hides this.

With --barge-in the microphone stays open while the agent speaks: start talking and it stops and
listens, with no key press. Its own voice from the speakers is told apart from yours (barge_in.py).
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import numpy as np
import sounddevice

from agent import VoiceAgent, add_agent_arguments
from barge_in import BLOCK, MIC_RATE_HZ, Duplex, EchoGate
from record import TRIM_END_SECONDS, TRIM_START_SECONDS, record_until_enter

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

# Shown as a fourth task with --barge-in: a long answer to talk over. The first word said over the
# agent overlaps its voice and is often misheard, so the line opens with a word that can be lost.
BARGE_IN_LINE = (
    "Where can I print, and how much is it?\n"
    "      then, while the agent is still answering, say over it (no key press):\n"
    "      Actually, what time does the cafe close on Saturdays?"
)
BARGE_IN_TASK = ("talking over the agent", BARGE_IN_LINE, (("closes", None), ("cents", BARGE_IN_LINE)))

CALIBRATION_PHRASE = "Hello. I am ready, and you can talk over me while I am speaking."
PRE_ROLL_SECONDS = 0.3  # kept from before the interruption was noticed
END_OF_SPEECH_SECONDS = 0.9  # this much quiet ends an utterance that interrupted the agent
LONGEST_UTTERANCE_SECONDS = 15

HELP = """
How this works
  1. Press Enter, say the line shown after SAY, wait a moment, press Enter again.
  2. The agent answers out loud. If it asks you something, the next SAY line is the answer.
  3. You do not have to give every detail at once: the agent asks for what is missing, one
     question at a time, and remembers what you already said.
  4. If it mishears you, answer its question in your own words, or type n + Enter to start the
     task again.
"""


def next_line(task: tuple, reply: str) -> tuple[str | None, bool]:
    """Return (what to say next, whether the task is complete) after the agent's reply."""
    for words, line in task[2]:
        if words.lower() in reply.lower():
            return line, line is None

    return None, False


def as_pcm(blocks: list[np.ndarray]) -> bytes:
    return np.concatenate(blocks).astype(np.int16).tobytes() if blocks else b""


class PushToTalk:
    """The microphone is open only between two presses of Enter; the agent cannot be interrupted."""

    def __init__(self, agent: VoiceAgent) -> None:
        self.agent = agent
        self.speaker = sounddevice.RawOutputStream(samplerate=agent.sample_rate_hz, channels=1, dtype="int16")

    def __enter__(self) -> "PushToTalk":
        self.speaker.start()

        return self

    def __exit__(self, *_error) -> None:
        self.speaker.stop()

    def listen(self) -> bytes:
        return record_until_enter("listening... [Enter when done] ")

    def answer(self, pcm: bytes) -> tuple[dict, float | None, bytes | None]:
        """Return the agent's extras, when its first audio was played, and no interruption."""
        first_audio = []

        def play(chunk: bytes) -> None:
            first_audio.append(time.monotonic())
            self.speaker.write(chunk)

        extras = self.agent.respond(pcm, lambda event: None, play)

        return extras, first_audio[0] if first_audio else None, None


class BargeIn:
    """The microphone stays open while the agent speaks, and speech from the user stops it."""

    def __init__(self, agent: VoiceAgent) -> None:
        self.agent = agent
        self.io = Duplex(agent.sample_rate_hz)
        self.gate = EchoGate()
        self.speech_level = 250.0

    def __enter__(self) -> "BargeIn":
        self.io.__enter__()
        print("\nSetting up barge-in: please stay quiet while I say one sentence.")
        time.sleep(0.3)
        self.io.drain()
        quiet = []

        while len(quiet) * BLOCK < 0.6 * MIC_RATE_HZ:
            if (block := self.io.next_block()) is not None:
                quiet.append(block[0])

        for chunk in self.agent.voice.synthesize(CALIBRATION_PHRASE):
            self.io.play(chunk.audio_int16_bytes)

        # What the microphone hears while only the agent speaks, with a moment for the room to settle.
        pairs, settled = [], 0

        while settled * BLOCK < 0.4 * MIC_RATE_HZ:
            if (block := self.io.next_block()) is not None:
                pairs.append(block)
                settled = 0 if self.io.playing else settled + 1

        self.gate.calibrate(quiet, pairs)
        noise = float(np.median([np.sqrt(np.mean(block**2)) for block in quiet]))
        self.speech_level = max(3 * noise, 250.0)

        return self

    def __exit__(self, *error) -> None:
        self.io.__exit__(*error)

    def listen(self) -> bytes:
        self.io.drain()
        input("listening... [Enter when done] ")
        pcm = as_pcm(self.io.drain())
        first = int(TRIM_START_SECONDS * MIC_RATE_HZ) * 2
        last = len(pcm) - int(TRIM_END_SECONDS * MIC_RATE_HZ) * 2

        return pcm[first:last]

    def answer(self, pcm: bytes) -> tuple[dict, float | None, bytes | None]:
        """Return the agent's extras, when its first audio was played, and what the user said over
        the agent, if they did."""
        interrupted = threading.Event()
        first_audio, outcome = [], {}

        def play(chunk: bytes) -> None:
            # After an interruption the rest of the reply is still generated, but no longer played.
            if not interrupted.is_set():
                first_audio.append(time.monotonic())
                self.io.play(chunk)

        def respond() -> None:
            try:
                outcome["extras"] = self.agent.respond(pcm, lambda event: None, play)
            except Exception as error:  # noqa: BLE001
                outcome["error"] = error

        worker = threading.Thread(target=respond, daemon=True)
        self.io.drain()
        self.gate.start()
        worker.start()

        recent = deque(maxlen=round(PRE_ROLL_SECONDS * MIC_RATE_HZ / BLOCK))
        settled, said = 0, None

        while settled * BLOCK < 0.3 * MIC_RATE_HZ:
            if (block := self.io.next_block()) is None:
                continue

            recent.append(block[0])
            settled = 0 if worker.is_alive() or self.io.playing else settled + 1

            if self.gate.heard_user(*block):
                interrupted.set()
                self.io.flush()
                said = as_pcm(list(recent) + self._until_quiet())
                break

        worker.join()

        if "error" in outcome:
            raise outcome["error"]

        return outcome["extras"], first_audio[0] if first_audio else None, said

    def _until_quiet(self) -> list[np.ndarray]:
        """The microphone blocks up to the end of the user's utterance."""
        blocks, quiet = [], 0

        while quiet * BLOCK < END_OF_SPEECH_SECONDS * MIC_RATE_HZ and len(blocks) * BLOCK < LONGEST_UTTERANCE_SECONDS * MIC_RATE_HZ:
            if (block := self.io.next_block()) is None:
                continue

            blocks.append(block[0])
            quiet = quiet + 1 if np.sqrt(np.mean(block[0] ** 2)) < self.speech_level else 0

        return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_agent_arguments(parser)
    parser.add_argument("--no-guide", action="store_true", help="do not show what to say")
    parser.add_argument("--barge-in", action="store_true", help="let the user talk over the agent to interrupt it")
    arguments = parser.parse_args()
    agent = VoiceAgent(arguments)
    agent.warm_up()

    # The tasks to show, the one being shown (None when no guide is wanted or left), the line to say.
    guide = GUIDE + ((BARGE_IN_TASK,) if arguments.barge_in else ())
    task = None if arguments.no_guide else 0
    line = None if task is None else guide[0][1]

    with (BargeIn if arguments.barge_in else PushToTalk)(agent) as audio:
        if task is not None:
            print(HELP)
            print(f"Task 1 of {len(guide)}: {guide[0][0]}.")

        said_over = None  # what the user said over the agent, to be answered next

        while True:
            if said_over is None:
                if line:
                    print(f"\nSAY:  {line}")

                command = input("[Enter to talk, n = new conversation, q = quit] ").strip().lower()

                if command == "q":
                    break

                if command == "n":
                    agent.reset()
                    print("New conversation.")
                    line = None if task is None else guide[task][1]
                    continue

                pcm = audio.listen()

            else:
                pcm = said_over

            released = time.monotonic()
            extras, first_audio, said_over = audio.answer(pcm)
            delay = f"{(first_audio - released) * 1000:.0f} ms to first audio" if first_audio else "no audio"
            print(f"you:   {extras['transcript']}\nagent: {extras['reply']}\n({delay})")

            if said_over is not None:
                print("-- you talked over the agent: it stopped and is answering that now --")
                continue

            if task is None:
                continue

            line, complete = next_line(guide[task], extras["reply"])

            if complete:
                # Each task is its own conversation, as in the recorded dialogues.
                agent.reset()
                task = task + 1 if task + 1 < len(guide) else None

                if task is None:
                    print(f"\nAll {len(guide)} tasks shown. Type q + Enter to quit.")
                else:
                    line = guide[task][1]
                    print(f"\nDone. Task {task + 1} of {len(guide)}: {guide[task][0]}.")

            elif line is None:
                print("\nNo suggestion for that reply: answer the agent in your own words, or type n + Enter to redo the task.")


if __name__ == "__main__":
    main()
