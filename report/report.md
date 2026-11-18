# Assignment 4 report: Hartwell voice agent

> DRAFT. Every number below is from `run_output/7153569/trace.jsonl` or the run folders named next to
> it. Lines marked **WRITE** are the analysis the course requires you to write yourselves; the
> evidence for each is listed under it. Delete every line that starts with ">" before submitting.

**Partners:** Tasos Siris, asiris3, and Muyu [last name, Georgia Tech username]  
**Models:** LLM Llama 3.1 8B Instruct (vLLM, bfloat16, temperature 0); ASR faster-whisper `small.en` (float16, beam 5); TTS Piper `en_US-lessac-medium`  
**Run behind the figures:** PACE-ICE job 7153569; 8 CPU cores [CPU model] and one NVIDIA H100 80GB; 40 turns; trace file `run_output/7153569/trace.jsonl`

## 1. The system

The agent is a cascade: faster-whisper transcribes the whole utterance once the speaker stops; the
LLM is called once, not streamed, to pick the intent as JSON (never spoken); rules read numbers,
times, days and equipment from the transcript; deterministic task logic keeps the timers, reminders
and booking and words the reply; the LLM is called a second time, streamed, to speak that reply (or,
for a lookup, to answer from the knowledge base); Piper synthesizes the stream piece by piece.
ASR and the intent call wait for the stage before them; the second LLM call and the TTS overlap.
The stream is cut for the TTS at sentence ends, and the first piece also at the first comma or
semicolon. The whole knowledge base (one JSON file) is in the prompt of the second call on lookup
turns only; the intent prompt holds only the room and feature names. Whisper and the LLM run on the
GPU, Piper on the CPU; CTranslate2 and ONNX Runtime are both set to the job's 8 cores. Before the
first timed turn the agent runs every stage twice with both prompts on two seconds of noise.

## 2. The latency waterfall

![Latency waterfall](waterfall.png)

| | p50 (ms) | p95 (ms) |
|---|---|---|
| **TTFA, all turns** | 235 | 342 |
| ASR: `asr_final` minus `end_of_user_speech` | 36 | 44 |
| LLM start: `llm_first_token` minus `asr_final` | 91 | 122 |
| Speech start: `tts_first_audio` minus `llm_first_token` | 100 | 225 |
| Not on the TTFA path: `llm_complete` minus `llm_first_token` | 64 | 208 |

Turns counted: 40 of 40.

Each segment of the figure is that stage's own percentile over the 40 turns, so the three p95
segments add up to more than the p95 TTFA; the black line is the TTFA percentile measured over the
turns. The agent's extra stamps cut the two LLM-side stages further (p50 / p95, ms): intent call
78 / 109; first token of the spoken reply 14 / 20; first piece of text complete 49 / 103; Piper
synthesis of the first piece 56 / 133.

**How I treat stages that overlap:** `llm_first_token` is the first token of the second, spoken LLM
call, so the silent intent call is counted in "LLM start". Piper starts on the first piece while the
LLM is still streaming; the wait for that first piece is counted in "Speech start", and the rest of
the stream (`llm_complete` minus `llm_first_token`) is reported but is not on the path to first audio.

**How to recompute these numbers:** `python3 waterfall.py --trace run_output/7153569/trace.jsonl --figure report/waterfall.png`

## 3. Word error rate on our own voices

| Speaker | Utterances | Reference words | Pooled WER | Mean WER per utterance |
|---|---|---|---|---|
| Tasos | 20 | 118 | 3.4% | 2.1% |
| Muyu | 20 | 118 | 21.2% | 22.4% |

> **WRITE** (two or three sentences): which voice the ASR handles worse, what the difference looks
> like, one quoted utterance, and anything about the recording that could explain it. Evidence:
> - Both speakers read the same 20 sentences, so the gap is not the wording.
> - Tasos: 4 wrong words, 2 utterances. "set a timer for five minutes for the eggs" was heard as
>   "sit the timer for five minutes for the egg"; "can i get a room" as "can we get a room".
> - Muyu: 25 wrong words. "cancel the eggs timer" was heard as "console the x timer"; "five" as
>   "fife"; "set another timer for twenty minutes for the laundry" as "set another timer over ten
>   minutes for"; "tea" as "table"; "make it" as "mix it"; "Thursday afternoon" as "Thursday after this".
> - Recording facts: Muyu's takes peak at about 2,300 to 5,500 of 32,767, Tasos's at about 5,800 to
>   12,600; several of Muyu's files end within 0.1 s of the last word (for example `muyu_d03_t03.wav`),
>   and the laundry sentence lost its last two words.
> - Effect on tasks: all 6 of Tasos's dialogues completed; 2 of Muyu's 6 did (reminder, lookup).
>   The failures follow the misheard word (the timer was labelled "table" and "axe").

## 4. The bottleneck

> **WRITE** all four items yourselves. This section is graded on whether the evidence supports the
> claim. Evidence you can draw on:

**The dominant stage:** [WRITE]

> - Speech start: p50 100 ms of a 235 ms TTFA (43%). LLM start: 91 ms (39%). ASR: 36 ms (15%).
> - Inside speech start at the median: 49 ms waiting for the first piece of text, 56 ms of Piper.
> - Inside LLM start: the intent call is 78 ms of the 91 ms.

**Why:** [WRITE]

> - Piper runs on the CPU (8 threads); Whisper and the LLM run on the H100.
> - Turns where the task logic words the reply (34 turns): first piece ready after 37 ms, Piper 55 ms,
>   TTFA p50 234 ms. Lookup turns answered by the LLM (6 turns): first piece after 73 ms, Piper 88 ms,
>   TTFA p50 277 ms.
> - The three slowest turns (341, 362, 402 ms) are the long room proposals: their first piece took
>   61 to 103 ms to arrive and 122 to 166 ms to synthesize.
> - Fastest turn 152 ms, slowest 402 ms.

**The fix I would ship next:** [WRITE: one change, the expected effect on the p50, and the reasoning]

> - Run `run_output/7153831/` (first piece not cut at a comma): TTFA p50 253 ms against 235 ms, speech
>   start 119 ms against 100 ms. A shorter first piece is measurably faster.
> - Run `run_output/7153832/` (Llama 3.2 3B): TTFA p50 213 ms, intent call 66 ms against 78 ms, first
>   piece 28 ms against 49 ms; but it said "fourteen hundred" for the cafe's closing time and took two
>   misheard lines for room names.
> - Not measured, so only as reasoning: for timer, reminder and booking turns the reply text exists
>   before the second LLM call starts; Piper on the GPU; a shorter first piece.

**One thing I tried that did not help:** [WRITE]

> - Run `run_output/7153830/` (Whisper beam size 1 instead of 5): ASR p50 34 ms against 36 ms, TTFA p50
>   230 ms against 235 ms, p95 323 against 342. WER rose from 3.4% to 5.9% (Tasos) and from 21.2% to
>   24.6% (Muyu). Each configuration was run once, so a difference of a few milliseconds is within
>   run-to-run variation.

## 5. Reproducing this

From the unzipped submission on PACE-ICE (commands also in `README.md`):

```bash
export SCRATCH_DIR=$HOME/scratch
export APPTAINER_CACHEDIR=$SCRATCH_DIR/cache APPTAINER_TMPDIR=$SCRATCH_DIR/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def   # about 4 minutes
sbatch run_agent.sbatch                                                 # about 4 minutes
```

On a login node without Apptainer, prefix the build with
`srun -p ice-cpu --cpus-per-task=8 --mem=32G --time=00:40:00`. The job plays the 40 recordings under
`recordings/` and writes `run_output/<job id>/trace.jsonl`, the spoken replies under `tts/`, and the
tables and WER in `a4-agent-<job id>.out`. Dependencies are pinned in `recipe/requirements.txt`. The
LLM runs at temperature 0 with seed 0 (`SEED` in `agent.py`), vLLM is started with `--seed 42`, and
Whisper decodes with beam search and no sampling.

## Appendix: AI-use disclosure

> **WRITE** this yourselves; it is graded for candor. The facts:

**Tools used:** Claude Code (Anthropic), model Claude Fable 5.2. [Add anything else either of you used.]

**What for:**
> - It wrote the code: `agent.py`, `dialogue.py`, `nlu.py`, `waterfall.py`, `record.py`, `demo.py`,
>   the tests, the dialogue scripts, `recipe/requirements.txt`, the changes to `run_agent.sbatch`,
>   and the README.
> - It built the image and submitted the jobs on PACE-ICE, copied the results back and drew the figure.
> - In this report it filled in the header, section 1, the tables, the overlap paragraph and
>   section 5, and listed the measurements under sections 3 and 4.
> - A design fact a reader should know: for timers, reminders and bookings the reply text is produced
>   by code and the LLM repeats it (word for word in 34 of 34 such turns in the reported run).

**What I wrote or decided myself:** [WRITE: the recordings, the analysis in sections 3 and 4, the
choice of what to ship, and whatever else is true.]

**How I checked it:**
> - 12 unit tests of the task logic against the example dialogues of `tasks/task_families.md`.
> - The course checker (`record_problems`) passes on all 40 turns of the shipped trace.
> - Every transcript and reply of the run was read (`run_output/7153569/job.out`); that reading found
>   and fixed an invented "fitness center" in a lookup answer and two misread numbers.
> - [Add what you checked yourselves, for example listening to the audio in `run_output/7153569/tts/`.]
