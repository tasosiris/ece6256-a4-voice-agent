"""Barge-in for the live demo: play the agent's speech, listen at the same time, and notice when
the user starts talking over it.

The microphone also hears the agent's own voice coming out of the speakers. EchoGate handles that
echo: from a phrase played at start-up it learns how strongly each frequency band of the played
sound arrives at the microphone, predicts the echo of whatever is being played, and reports speech
only when the microphone holds clearly more than that prediction.
"""

from __future__ import annotations

import queue
import threading
from collections import deque

import numpy as np
import sounddevice

MIC_RATE_HZ = 16_000
BLOCK = 512  # samples per microphone block, 32 ms

BAND_EDGES_HZ = (150, 300, 450, 600, 800, 1000, 1250, 1500, 1800, 2200, 2700, 3300, 4000, 5000, 6500, 7900)
FREQUENCIES = np.fft.rfftfreq(BLOCK, 1 / MIC_RATE_HZ)
BANDS = [np.where((FREQUENCIES >= low) & (FREQUENCIES < high))[0] for low, high in zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:])]
WINDOW = np.hanning(BLOCK)

# The settings below were chosen on 86 s of the agent's speech recorded through a laptop's own
# speakers and microphone, with our recorded utterances mixed in: no false alarm, 128 of 132
# utterances noticed, after 0.18 s at the median.
ECHO_MARGIN = 4.0  # the microphone must hold this many times the predicted echo power
TAIL_DECAY = 0.5  # how much of the echo is still in the room one block later
HITS_NEEDED, HITS_WINDOW = 4, 6  # blocks above the threshold among the last blocks
FLOOR_FACTOR = 30.0  # threshold as a multiple of the room noise
MINIMUM_THRESHOLD = 3e9  # band power of quiet speech, used when the room is silent


def band_power(block: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(block * WINDOW)) ** 2

    return np.array([spectrum[band].sum() for band in BANDS])


class EchoGate:
    """Decides, block by block, whether the microphone holds more than the agent's own echo."""

    def __init__(self) -> None:
        self.gain = np.zeros(len(BANDS))
        self.threshold = MINIMUM_THRESHOLD
        self.start()

    def start(self) -> None:
        """Forget the previous utterance."""
        # The echo reaches the microphone up to two blocks after the stream times say it was played.
        self.played = deque(maxlen=3)
        self.tail = np.zeros(len(BANDS))
        self.hits = deque(maxlen=HITS_WINDOW)

    def _echo_shape(self, played: np.ndarray) -> np.ndarray:
        """Power per band of the played sound as the room returns it, before the learned gain."""
        self.played.append(band_power(played))
        self.tail = np.maximum(np.max(self.played, axis=0), self.tail * TAIL_DECAY)

        return self.tail

    def calibrate(self, quiet: list[np.ndarray], pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Learn the room noise from silent blocks and the echo gain from (heard, played) blocks
        recorded while only the agent was speaking."""
        floor = float(np.median([band_power(block).sum() for block in quiet])) if quiet else 0.0
        self.threshold = max(FLOOR_FACTOR * floor, MINIMUM_THRESHOLD)

        self.start()
        heard = np.array([band_power(mic) for mic, _ in pairs])
        shape = np.array([self._echo_shape(played) for _, played in pairs])
        loud = shape.sum(axis=1) > np.percentile(shape.sum(axis=1), 40)

        if loud.any():
            self.gain = np.percentile(heard[loud] / np.maximum(shape[loud], 1e-9), 90, axis=0)

        self.start()

    def heard_user(self, mic: np.ndarray, played: np.ndarray) -> bool:
        excess = np.maximum(band_power(mic) - ECHO_MARGIN * self.gain * self._echo_shape(played), 0).sum()
        self.hits.append(excess > self.threshold)

        return sum(self.hits) >= HITS_NEEDED


class Duplex:
    """A speaker and a microphone that run together. Every microphone block comes with the sound
    that was being played during it, lined up by the times the two audio streams report."""

    RING = MIC_RATE_HZ * 30

    def __init__(self, speech_rate_hz: int) -> None:
        self.speech_rate_hz = speech_rate_hz
        self.pending = bytearray()
        self.lock = threading.Lock()
        self.blocks: queue.Queue[tuple[float, bytes]] = queue.Queue()
        self.reference = np.zeros(self.RING)
        self.origin: float | None = None
        self.output = sounddevice.RawOutputStream(
            samplerate=speech_rate_hz, channels=1, dtype="int16", blocksize=BLOCK, callback=self._fill
        )
        self.input = sounddevice.RawInputStream(
            samplerate=MIC_RATE_HZ, channels=1, dtype="int16", blocksize=BLOCK,
            callback=lambda data, _frames, when, _status: self.blocks.put((when.inputBufferAdcTime, bytes(data))),
        )

    def __enter__(self) -> "Duplex":
        self.output.start()
        self.input.start()

        return self

    def __exit__(self, *_error) -> None:
        self.input.stop()
        self.output.stop()

    def _fill(self, outdata, frames: int, when, _status) -> None:
        with self.lock:
            chunk = bytes(self.pending[: frames * 2])
            del self.pending[: frames * 2]

        chunk += b"\0" * (frames * 2 - len(chunk))
        outdata[:] = chunk

        if self.origin is None:
            self.origin = when.outputBufferDacTime

        # Keep what is played at the microphone's rate, at the time it leaves the speaker.
        played = np.frombuffer(chunk, dtype=np.int16).astype(float)
        count = round(frames * MIC_RATE_HZ / self.speech_rate_hz)
        resampled = np.interp(np.arange(count) * self.speech_rate_hz / MIC_RATE_HZ, np.arange(frames), played)
        start = round((when.outputBufferDacTime - self.origin) * MIC_RATE_HZ)
        self.reference[(start + np.arange(count)) % self.RING] = resampled

    def play(self, pcm: bytes) -> None:
        with self.lock:
            self.pending += pcm

    def flush(self) -> None:
        """Stop speaking now: drop everything that has not reached the speaker yet."""
        with self.lock:
            self.pending.clear()

    @property
    def playing(self) -> bool:
        return bool(self.pending)

    def next_block(self, timeout: float = 0.2) -> tuple[np.ndarray, np.ndarray] | None:
        """The next microphone block and what was played during it, or None after the timeout."""
        try:
            when, data = self.blocks.get(timeout=timeout)
        except queue.Empty:
            return None

        mic = np.frombuffer(data, dtype=np.int16).astype(float)

        if self.origin is None or len(mic) != BLOCK:
            return mic, np.zeros(len(mic))

        start = round((when - self.origin) * MIC_RATE_HZ)

        return mic, self.reference[(start + np.arange(BLOCK)) % self.RING]

    def drain(self) -> list[np.ndarray]:
        """All microphone blocks that are waiting, without their played sound."""
        blocks = []

        while (block := self.next_block(timeout=0)) is not None:
            blocks.append(block[0])

        return blocks
