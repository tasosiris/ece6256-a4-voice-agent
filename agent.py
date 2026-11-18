"""Push-to-talk voice agent: faster-whisper -> a served LLM -> Piper, measured through the audio shim.

One turn: the whole utterance is transcribed when the speaker stops; one LLM call turns the
transcript into an action that dialogue.py carries out (never spoken); a second, streamed LLM call
speaks the result, and each sentence goes to Piper as soon as it is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import onnxruntime
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import PiperConfig

from dialogue import ACTION_SCHEMA, Dialogue
from voice_kit import ASR_FINAL, LLM_COMPLETE, LLM_FIRST_TOKEN, ReplaySession, now_nanoseconds

REPOSITORY = Path(__file__).resolve().parent
INPUT_SAMPLE_RATE_HZ = 16_000
HISTORY_EXCHANGES = 4
SEED = 0

ROUTER_PROMPT = """You label what a user said to the voice assistant of the Hartwell Community Center. Output one JSON object and nothing else.

"intent" is one of:
- timer_set: start a timer.
- timer_change: change the length of a running timer, or add time to it.
- timer_check: how much time is left on a timer.
- timer_cancel: stop a timer.
- reminder_set: set or change a reminder at a time of day.
- reminder_cancel: remove a reminder.
- list: which timers or reminders exist.
- booking: anything about finding, choosing, changing, accepting or declining a meeting room, including a short answer to the assistant's own question about a room.
- booking_cancel: cancel a room booking that was already made.
- lookup: a question about the center: opening hours, where something is, prices, rules, what a room has.
- other: anything else.

Optional fields, copied from the user's own words and left out when the user did not say them:
- "label": what a timer is for ("tea"), or what to be reminded of as a verb phrase ("water the plants").
- "name": the person's name to book a room under.
- "room": a room the user names.
- "features": equipment or qualities the user asks for in a room.

Examples:
"Start a ten minute timer for the tea." -> {"intent":"timer_set","label":"tea"}
"Actually make it fifteen." -> {"intent":"timer_change"}
"How long is left?" -> {"intent":"timer_check"}
"Cancel the eggs timer." -> {"intent":"timer_cancel","label":"eggs"}
"Remind me to water the plants." -> {"intent":"reminder_set","label":"water the plants"}
After the assistant asked "At what time?": "Six in the evening." -> {"intent":"reminder_set"}
"What reminders do I have?" -> {"intent":"list"}
"I need a room on Thursday afternoon." -> {"intent":"booking"}
After the assistant asked "How many people?": "Five." -> {"intent":"booking"}
"We need a projector and a piano." -> {"intent":"booking","features":["projector","piano"]}
"Maple Hall is fine. Wednesday at two." -> {"intent":"booking","room":"Maple Hall"}
"Yes, under the name Sam." -> {"intent":"booking","name":"Sam"}
"Where can I print, and how much is it?" -> {"intent":"lookup"}
"Which rooms have a projector?" -> {"intent":"lookup"}"""

VOICE_STYLE = (
    "You are the voice assistant of the Hartwell Community Center. The user hears your reply, so use "
    "one or two short plain sentences, with no lists, symbols or digits: write numbers and times as "
    "words, such as 'two in the afternoon' or 'ten cents'."
)

# A request that changes state is carried out by dialogue.py, which words its own result; the LLM
# only has to say it. Letting it rephrase cost facts ("How many people?" came back as an answer).
RELAY_PROMPT = (
    "You are the voice of an assistant. Each user message is the exact text to speak. Repeat it word "
    "for word, with nothing before or after it. Never answer it, even when it is a question."
)

# Without examples the model answers a reply that is itself a question ("How many people?").
RELAY_EXAMPLES = (
    "How many people?",
    "Ten minutes for the tea, starting now.",
    "Booked. Elm Room, Monday at one in the afternoon, for Dana.",
    "Birch Room on the first floor is free on Friday at ten in the morning. It holds four. Shall I book it?",
)

ANSWER_PROMPT = (
    VOICE_STYLE + " Answer only from the knowledge base below. Never guess: if it does not hold the "
    "answer, do not say yes or no: say only that you don't have anything about it, and name "
    "nothing else the center might have. "
    "When a question has two parts, answer both. Timers, reminders and room bookings are handled "
    "elsewhere, so never claim to have set, changed or booked anything; if the user seems to want "
    "that, ask them to say it again.\n\nKnowledge base:\n"
)

SENTENCE_END = re.compile(r"[.!?][\"')]?\s")
CLAUSE_END = re.compile(r"[,;]\s")


def job_cpu_count() -> int:
    """CPUs this job may use; a cluster node has many more than the job was given."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    return os.cpu_count() or 1


def natural_key(path: Path) -> list:
    """Sort key that puts turn_2 before turn_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def dialogue_key(path: Path) -> str:
    """Recordings whose names differ only in a final number are turns of one conversation."""
    return str(path.parent / re.sub(r"\d+$", "", path.stem))


class ChatClient:
    """The served LLM, through the OpenAI-compatible chat API of vLLM."""

    def __init__(self, base_url: str, model: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.schema_supported = True

    def _open(self, messages: list[dict], max_tokens: int, **options):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": SEED,
            **options,
        }
        request = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )

        return urllib.request.urlopen(request, timeout=120)

    def complete_json(self, messages: list[dict], schema: dict, max_tokens: int = 80) -> dict:
        """Return the JSON object the model wrote under the schema, or {} if it is not valid JSON."""
        options = {}

        if self.schema_supported:
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "schema": schema},
            }

        try:
            with self._open(messages, max_tokens, **options) as response:
                text = json.load(response)["choices"][0]["message"]["content"] or ""

        except urllib.error.HTTPError as error:
            if not options or error.code != 400:
                raise

            # A server without schema-constrained output: the prompt alone still asks for JSON.
            self.schema_supported = False
            return self.complete_json(messages, schema, max_tokens)

        try:
            value = json.loads(text[text.index("{") : text.rindex("}") + 1])
        except ValueError:
            return {}

        return value if isinstance(value, dict) else {}

    def stream(self, messages: list[dict], max_tokens: int = 120) -> Iterator[str]:
        """Yield the pieces of the answer as the server sends them."""
        with self._open(messages, max_tokens, stream=True) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue

                data = line[5:].strip()

                if data == b"[DONE]":
                    break

                choices = json.loads(data).get("choices") or [{}]
                piece = choices[0].get("delta", {}).get("content")

                if piece:
                    yield piece


def find_voice(name: str) -> tuple[Path, Path]:
    """Return the model and config files of a Piper voice: a path, or a voice of rhasspy/piper-voices
    in the Hugging Face cache (HF_HOME), wherever the cache keeps it."""
    if Path(name).is_file():
        return Path(name), Path(f"{name}.json")

    from huggingface_hub import constants, hf_hub_download

    language, speaker, quality = name.split("-")
    folder = f"{language.split('_')[0]}/{language}/{speaker}/{quality}"

    try:
        model = hf_hub_download("rhasspy/piper-voices", f"{folder}/{name}.onnx")
        return Path(model), Path(hf_hub_download("rhasspy/piper-voices", f"{folder}/{name}.onnx.json"))

    except Exception:  # noqa: BLE001
        # The shared cache may hold the voice under another repository or as plain files.
        for model in Path(constants.HF_HOME).rglob(f"{name}.onnx"):
            if Path(f"{model}.json").is_file():
                return model, Path(f"{model}.json")

    raise FileNotFoundError(
        f"Piper voice {name!r} is not in {constants.HF_HOME}; pass --tts-voice /path/to/voice.onnx"
    )


class VoiceAgent:
    """The three models and one conversation. `respond` answers one utterance."""

    def __init__(self, settings: argparse.Namespace) -> None:
        threads = settings.cpu_count or job_cpu_count()

        import ctranslate2

        device = settings.asr_device

        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"

        # CTranslate2 and ONNX Runtime size their thread pools from the machine unless told otherwise.
        self.asr = WhisperModel(
            settings.asr_model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            cpu_threads=threads,
        )
        self.asr_beam_size = settings.asr_beam_size

        model_path, config_path = find_voice(settings.tts_voice)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = threads
        self.voice = PiperVoice(
            config=PiperConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8"))),
            session=onnxruntime.InferenceSession(
                str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
            ),
        )
        self.sample_rate_hz = self.voice.config.sample_rate

        self.llm = ChatClient(settings.llm_base_url, settings.llm_model)
        self.kb_path = Path(settings.knowledge_base)
        self.split_first_clause = not settings.no_clause_split
        self.reset()
        self.answer_prompt = ANSWER_PROMPT + self.dialogue.knowledge_text()

        # Whisper spells the venue's own names right when it has seen them.
        kb = self.dialogue.kb
        names = [kb["venue"]["name"]] + [entry["name"] for entry in kb["areas"] + kb["rooms"]]
        self.asr_hotwords = ", ".join(names)

        print(
            f"agent: asr {settings.asr_model} on {device}, tts {model_path.name} at "
            f"{self.sample_rate_hz} Hz, llm {settings.llm_model}, {threads} threads",
            flush=True,
        )

    def reset(self) -> None:
        """Start a new conversation."""
        self.dialogue = Dialogue(self.kb_path)
        self.history: list[tuple[str, str]] = []

    def warm_up(self) -> None:
        """Run every stage, with both prompts, so no start-up cost lands inside a timed turn."""
        noise = np.random.default_rng(SEED).normal(0, 300, 2 * INPUT_SAMPLE_RATE_HZ).astype(np.int16)

        for _ in range(2):
            self.transcribe(noise.tobytes())
            self.route("What time does the cafe close?")
            "".join(self.llm.stream(self.speech_messages("Hello.", "Hello."), max_tokens=8))
            "".join(self.llm.stream(self.speech_messages("Hello.", None), max_tokens=8))

            for _chunk in self.voice.synthesize("The agent is ready."):
                pass

    def transcribe(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.asr.transcribe(
            audio,
            language="en",
            beam_size=self.asr_beam_size,
            hotwords=self.asr_hotwords,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        return " ".join(segment.text.strip() for segment in segments).strip()

    def route(self, transcript: str) -> dict:
        """Ask the LLM which action the user wants. Its output is never spoken."""
        recent = "\n".join(f"User: {user}\nAssistant: {reply}" for user, reply in self.history)
        content = (
            f"State:\n{self.dialogue.summary()}\n\nRecent conversation:\n{recent or '(none)'}\n\n"
            f'Latest user message: "{transcript}"'
        )
        messages = [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": content}]

        return self.llm.complete_json(messages, ACTION_SCHEMA)

    def speech_messages(self, transcript: str, result: str | None) -> list[dict]:
        """The prompt of the spoken reply: relay a result, or answer from the knowledge base."""
        if result is not None:
            messages = [{"role": "system", "content": RELAY_PROMPT}]

            for example in RELAY_EXAMPLES:
                messages += [{"role": "user", "content": example}, {"role": "assistant", "content": example}]

            return messages + [{"role": "user", "content": result}]

        messages = [{"role": "system", "content": self.answer_prompt}]

        for user, reply in self.history:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]

        return messages + [{"role": "user", "content": transcript}]

    def respond(
        self, pcm: bytes, stamp: Callable[[str], None], write_audio: Callable[[bytes], None]
    ) -> dict:
        """Answer one utterance: stamp the three agent events, send speech to write_audio, and
        return the extras for the trace."""
        transcript = self.transcribe(pcm)
        stamp(ASR_FINAL)

        if transcript:
            action = self.dialogue.interpret(transcript, self.route(transcript))
            result = self.dialogue.handle(action)
        else:
            action, result = {}, "Sorry, I did not catch that. Could you say it again?"

        extras = {
            "transcript": transcript,
            "action": action,
            "audio_seconds": len(pcm) / (2 * INPUT_SAMPLE_RATE_HZ),
            "action_done_nanoseconds": now_nanoseconds(),
        }

        segments: queue.Queue[str | None] = queue.Queue()
        speech_errors = []

        def speak() -> None:
            try:
                for text in iter(segments.get, None):
                    for chunk in self.voice.synthesize(text):
                        write_audio(chunk.audio_int16_bytes)

            except Exception as error:  # noqa: BLE001
                speech_errors.append(error)

        speaker = threading.Thread(target=speak, daemon=True)
        speaker.start()

        reply, unspoken = "", ""

        try:
            for piece in self.llm.stream(self.speech_messages(transcript, result)):
                if not reply:
                    stamp(LLM_FIRST_TOKEN)

                reply += piece
                unspoken += piece

                while cut := self._segment_end(unspoken, first="first_segment_nanoseconds" not in extras):
                    extras.setdefault("first_segment_nanoseconds", now_nanoseconds())
                    segments.put(unspoken[:cut].strip())
                    unspoken = unspoken[cut:]

            stamp(LLM_COMPLETE)

            if unspoken.strip():
                extras.setdefault("first_segment_nanoseconds", now_nanoseconds())
                segments.put(unspoken.strip())

        finally:
            segments.put(None)
            speaker.join()

        if speech_errors:
            raise speech_errors[0]

        reply = reply.strip()
        self.history = (self.history + [(transcript, reply)])[-HISTORY_EXCHANGES:]
        extras["reply"] = reply

        if result is not None:
            # Whether the LLM spoke the task logic's reply word for word.
            extras["relay_exact"] = reply == result

        return extras

    def _segment_end(self, text: str, first: bool) -> int:
        """Where the next piece for the TTS ends: a sentence, or for the very first piece a clause
        of three words or more, so that speech can start before the first sentence is finished."""
        sentence = SENTENCE_END.search(text)

        if first and self.split_first_clause:
            clause = CLAUSE_END.search(text)

            if clause and len(text[: clause.start()].split()) >= 3:
                if not sentence or clause.end() < sentence.end():
                    return clause.end()

        return sentence.end() if sentence else 0


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-model", required=True, help="model name the server serves")
    parser.add_argument("--llm-base-url", required=True, help="for example http://127.0.0.1:8000/v1")
    parser.add_argument("--cpu-count", type=int, default=0, help="threads per library; 0 uses the job's CPUs")
    parser.add_argument("--asr-model", default="small.en", help="faster-whisper size or model folder")
    parser.add_argument("--asr-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--asr-beam-size", type=int, default=5)
    parser.add_argument("--tts-voice", default="en_US-lessac-medium", help="Piper voice name or .onnx path")
    parser.add_argument("--knowledge-base", default=str(REPOSITORY / "toy_kb" / "knowledge_base.json"))
    parser.add_argument(
        "--no-clause-split", action="store_true", help="send whole sentences only to the TTS"
    )


def run_session(settings: argparse.Namespace) -> None:
    """Replay every recording as one turn and write the trace."""
    recordings = sorted(Path(settings.recordings_directory).rglob("*.wav"), key=natural_key)

    if not recordings:
        sys.exit(f"No .wav files under {settings.recordings_directory}")

    # The shim appends to a trace, so a second run into the same folder would mix two runs.
    if (Path(settings.output_directory) / "trace.jsonl").exists():
        sys.exit(f"{settings.output_directory} already holds a trace; choose a new output directory")

    agent = VoiceAgent(settings)
    agent.warm_up()

    session = ReplaySession(
        recordings, Path(settings.output_directory), sink_sample_rate_hz=agent.sample_rate_hz
    )
    current_dialogue = None

    for turn in session.turns():
        key = dialogue_key(turn.recording.wav_path)

        if key != current_dialogue:
            agent.reset()
            current_dialogue = key

        received = bytearray()
        turn.start(on_chunk=received.extend)
        turn.end_of_speech_reached.wait()

        try:
            turn.extras.update(agent.respond(bytes(received), turn.stamp, turn.sink.write))

        except Exception as error:  # noqa: BLE001
            # The record is still written, without the missing events, and the checker rejects it.
            turn.extras["error"] = repr(error)

        turn.extras["dialogue"] = Path(key).name
        turn.finish()

        print(
            f"{turn.turn_id} [{turn.recording.wav_path.name}] "
            f"user: {turn.extras.get('transcript')!r} agent: {turn.extras.get('reply', turn.extras.get('error'))!r}",
            flush=True,
        )

    print(f"trace: {session.trace_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_agent_arguments(parser)
    parser.add_argument("--recordings-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    run_session(parser.parse_args())


if __name__ == "__main__":
    main()
