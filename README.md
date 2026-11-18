# Hartwell voice agent (ECE 6256 / CS 6604, Assignment 4)

A push-to-talk spoken agent for the three task families of `tasks/task_families.md` (timers and
reminders, lookup, two-constraint room booking), measured through the course audio shim. The handout
is in `ASSIGNMENT.md`.

```
recording ─▶ faster-whisper small.en ─▶ LLM call 1: intent (JSON, never spoken)
                                          │
                     rules read numbers, times, days, equipment (nlu.py)
                                          │
                     task logic: timers, reminders, room search (dialogue.py)
                                          │
              LLM call 2, streamed: the spoken reply ─▶ Piper, piece by piece ─▶ shim
```

- **ASR**: faster-whisper `small.en`, run once on the whole utterance when the speaker stops.
- **LLM**: any chat model behind an OpenAI-compatible server (vLLM on PACE-ICE). Default
  `meta-llama/Llama-3.1-8B-Instruct`. The model name and server URL are command-line settings.
- **TTS**: Piper `en_US-lessac-medium`, on the CPU. The reply is cut at sentence ends, and the first
  piece also at a comma, so speech starts before the LLM has finished.
- **`llm_first_token`** is the first token of call 2, the reply that is spoken. Call 1 produces no
  speech; its end is stamped in `extras.action_done_nanoseconds`, so its cost is visible.
- For a timer, reminder or booking, the reply text is produced by `dialogue.py` and the LLM repeats
  it. For a lookup, the LLM answers from the whole knowledge base in its prompt.

## Run on PACE-ICE

From the root of the repository, on a login node:

```bash
# 1. Build the image once (about 5 minutes). Keep it on scratch, not in your home directory.
export SCRATCH_DIR=$HOME/scratch
export APPTAINER_CACHEDIR=$SCRATCH_DIR/cache APPTAINER_TMPDIR=$SCRATCH_DIR/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def
# Some login nodes have no apptainer ("command not found"). Then run the same build on a compute node:
#   srun -p ice-cpu --cpus-per-task=8 --mem=32G --time=00:40:00 \
#       apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def

# 2. Run every .wav under recordings/ as one turn each.
sbatch run_agent.sbatch

# Your own recordings, another model, or agent options:
sbatch --export=ALL,RECORDINGS_DIR=my_dialogues run_agent.sbatch
sbatch --export=ALL,LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct,AGENT_ARGS="--asr-beam-size 1" run_agent.sbatch
```

The job serves the LLM with vLLM from the shared course environment, runs the agent offline against
the shared Hugging Face cache, and writes:

- `run_output/<job id>/trace.jsonl`: one record per turn
- `run_output/<job id>/tts/turn_NNNN.wav`: the speech the agent produced
- `a4-agent-<job id>.out`: every transcript and reply, the latency table, and the WER if
  `<recordings>/references.json` exists

The whole job takes about 6 minutes for the 40 recorded turns, most of it model loading.

### How recordings become conversations

Recordings are found recursively and played in natural order (`t2` before `t10`). Files whose names
differ only in a final number belong to one conversation, and the agent's memory is cleared when
that prefix changes: `sam_d01_t01.wav`, `sam_d01_t02.wav` are one dialogue, and `sam_d02_t01.wav`
starts a new one. To test your own dialogues, name the files `<anything>_d<dialogue>_t<turn>.wav`.

## Numbers for the report

```bash
python3 waterfall.py --trace run_output/<job id>/trace.jsonl --figure report/waterfall.png
python3 wer/score_wer.py --trace run_output/<job id>/trace.jsonl --references recordings/references.json
```

The run the report uses is `run_output/7153569/` (trace, spoken replies, and the job's log as
`job.out`), measured at the commit before the one that added it; `report/waterfall.png` is drawn from it.

Three more runs of the same 40 recordings, each changing one setting:

| Run | Change | Command |
|---|---|---|
| `run_output/7153830/` | Whisper beam size 1 | `sbatch --export=ALL,AGENT_ARGS="--asr-beam-size 1" run_agent.sbatch` |
| `run_output/7153831/` | first piece not cut at a comma | `sbatch --export=ALL,AGENT_ARGS="--no-clause-split" run_agent.sbatch` |
| `run_output/7153832/` | Llama 3.2 3B | `sbatch --export=ALL,LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct run_agent.sbatch` |

`waterfall.py` lists every turn the checker rejects, prints the p50 and p95 table of the report from
`voice_kit.summarize_stages`, splits the LLM-side stages with the agent's extra stamps, and draws
the figure. The table needs only the standard library; the figure needs matplotlib.

## Laptop: recording, live demo, tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt

python3 record.py --speaker <your name> --set a      # the partner records --set b
python3 -m unittest tests.test_dialogue               # task logic against the example dialogues
```

`record.py` shows each line of `dialogues/scripts.json`, records it at 16 kHz mono 16-bit into
`recordings/<speaker>/`, and writes the line into `recordings/references.json` as the reference
transcript. Sets `a` and `b` are 20 utterances each, 11 dialogues in all.

For a live conversation or a run through the shim on a laptop, serve a model locally, for example
with Ollama (`ollama pull llama3.1:8b`):

```bash
python3 demo.py  --llm-model llama3.1:8b --llm-base-url http://127.0.0.1:11434/v1
python3 agent.py --llm-model llama3.1:8b --llm-base-url http://127.0.0.1:11434/v1 \
    --recordings-directory recordings --output-directory run_output/laptop
```

The first run downloads the ASR and TTS weights from Hugging Face. Laptop timings are not the
reported ones.

## Reproducibility

- Dependencies are pinned in `recipe/requirements.txt` (the image) and `requirements-local.txt`.
- The LLM runs at temperature 0 with seed 0 (`SEED` in `agent.py`); vLLM is started with `--seed 42`.
  Whisper decodes with beam search and no sampling.
- Thread counts of CTranslate2 and ONNX Runtime are set from the CPUs the job was given
  (`--cpu-count`, or the scheduler affinity).
- Before the first timed turn the agent runs every stage twice with both prompts (`warm_up`).

## Files

```
agent.py            the agent: ASR, LLM client, TTS, the shim loop
dialogue.py         conversation state and task logic
nlu.py              rules that read values out of a transcript
waterfall.py        trace check, report tables, waterfall figure
record.py           record the scripted utterances
demo.py             live microphone conversation
dialogues/          the scripts that were recorded
recordings/         our recordings and their reference transcripts
run_agent.sbatch    the PACE-ICE job
recipe/             Apptainer definition and pinned requirements
tests/              tests of the task logic
voice_kit/, wer/, toy_kb/, tasks/, report_template/   the course kit, unchanged
```
