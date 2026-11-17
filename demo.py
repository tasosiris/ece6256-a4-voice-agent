"""Talk to the agent live with a microphone and speakers, for the demo video (not for measuring).

    python3 demo.py --llm-model llama3.1:8b --llm-base-url http://127.0.0.1:11434/v1

Push-to-talk with the Enter key: Enter to start talking, Enter when done. Type n + Enter to start a
new conversation and q + Enter to quit. Latency numbers come from the audio shim (agent.py), never
from this script.
"""

from __future__ import annotations

import argparse
import time

import sounddevice

from agent import VoiceAgent, add_agent_arguments
from record import record_until_enter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_agent_arguments(parser)
    agent = VoiceAgent(parser.parse_args())
    agent.warm_up()

    with sounddevice.RawOutputStream(samplerate=agent.sample_rate_hz, channels=1, dtype="int16") as speaker:
        while True:
            command = input("\n[Enter to talk, n = new conversation, q = quit] ").strip().lower()

            if command == "q":
                break

            if command == "n":
                agent.reset()
                print("New conversation.")
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


if __name__ == "__main__":
    main()
