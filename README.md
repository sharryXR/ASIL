# ASIL

ASIL (Agent-Software Interaction Layer) is the inference and evaluation
framework for replacing screenshot-and-click software control with structured
state observations and code-executable semantic actions.

This camera-ready release contains the runtime protocol, 15-application
benchmark adapters, GUI and native-interface baselines, independent raw-state
validation, Docker/Singularity environments, and the assisted software
onboarding framework described in the paper.

## Repository Scope

Included here:

- structured observation/action protocols and adapter runtime;
- adapters and action schemas for the 15 paper applications;
- 300 single-application, 80 multi-application, 80 hard, and 60 easy tasks;
- deterministic, agentic ASIL, CLI, screenshot GUI, UNO, and draw.io baselines;
- `asil.softwaregen` qualification, generation, audit, host probe, Docker probe,
  and evidence-report commands;
- Docker Compose and Singularity definitions, tests, and public release checks.

Training implementation, training-task generation, SFT demonstration
generation, RL rollout/training code, paper sources, rebuttal records, and
per-task evaluation trajectories or scores are intentionally not included.
Prepared datasets and released checkpoints are hosted on Hugging Face.

## Install

ASIL requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c constraints-host.txt -e ".[dev,eval]"
```

Run a no-key local smoke and the release inventory check:

```bash
pytest -q \
  tests/test_protocol.py \
  tests/test_softwaregen_examples.py \
  tests/test_raw_validation.py

python scripts/validate_public_release.py .
```

## Benchmark

Task-set indexes live in `evaluation_examples/`; each entry resolves to
`evaluation_examples/examples/<software>/<task_id>.json`.

| Index | Tasks | Purpose |
| --- | ---: | --- |
| `test_full15.json` | 300 | Single-application benchmark |
| `test_multi_apps_80.json` | 80 | Cross-application benchmark |
| `test_full15_multi_apps_380.json` | 380 | Main benchmark union |
| `test_full15_realwork_hard.json` | 80 | Held-out hard suite |
| `test_easy60.json` | 60 | Easier OSWorld-comparable band |

The 15 applications are Inkscape, LibreOffice Calc, LibreOffice Writer,
LibreOffice Impress, Blender, GIMP, OBS Studio, Gitea, code-server,
Thunderbird, Nautilus, Kdenlive, Audacity, draw.io, and JupyterLab.

Run deterministic evaluation:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --task-set test_full15.json \
  --participant asil \
  --asil-execution deterministic \
  --output-dir results/full15 \
  --output results/full15/results.json
```

Run an agentic ASIL participant after setting the provider key only in the
environment or an untracked `.env` file:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --task-set test_full15.json \
  --participant asil \
  --asil-execution agentic \
  --provider openai \
  --model gpt-5.4 \
  --max-steps 15 \
  --output-dir results/asil-agent \
  --output results/asil-agent/results.json
```

Set `ASIL_HYBRID_VISION=1` to supplement structured ASIL observations with the
adapter's rendered screenshot when a compatible multimodal endpoint is used.

The screenshot GUI participant uses the same task definitions and evaluator:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --task-set test_easy60.json \
  --participant gui \
  --provider openai \
  --model gpt-5.4 \
  --max-steps 50
```

Matched application-native baselines are available in
`scripts/run_uno_baseline.py` and `scripts/run_drawio_baseline.py`.

## Docker

The clean-device bootstrap builds the local eval and OBS images, starts the
pinned service stack, runs a deterministic four-task smoke, validates runtime
capabilities, and removes its dedicated Compose resources. No model API key is
required.

```bash
bash scripts/bootstrap_rebuttal_docker.sh
```

The historical script filename is retained for command compatibility; its
inventory now checks only this public release. See `docs/docker-quickstart.md`.
Singularity definitions and runner instructions are in
`docs/singularity-backend.md`.

## Assisted Software Onboarding

`asil.softwaregen` turns reviewed evidence about an open file, native script,
command, or HTTP interface into a declarative ASIL extension candidate. Runtime
hosts and executable allowlists come from the reviewed profile, not the model.
Generated bundles must pass static audit before any probe; actions require both
an explicit action file and `--allow-actions`.

```bash
PYTHONPATH=src python -m asil.softwaregen catalog
PYTHONPATH=src python -m asil.softwaregen qualify \
  examples/softwaregen/gitea_profile.json

PYTHONPATH=src python -m asil.softwaregen generate \
  examples/softwaregen/gitea_profile.json \
  --provider openai \
  --model gpt-5.4 \
  --output results/softwaregen/gitea

PYTHONPATH=src python -m asil.softwaregen audit \
  results/softwaregen/gitea/extension.json
```

The released Gitea measurement starts after a 97-line evidence profile already
exists. It used one model call and 24.832775 seconds for the post-profile
generation stage, produced one observation view and two semantic operations,
passed static audit with zero errors and warnings, and passed 3/3 host plus 3/3
Docker probes with repository-count restoration `2 -> 3 -> 2`.

These measurements do not include interface discovery, profile authoring,
custom bridge work, task/evaluator design, GUI synchronization, rendering, or
total historical person-hours. The 15 existing adapters predate this generator;
the catalog shows recurring access paths, not retrospective generation
provenance. Sanitized evidence is under `examples/softwaregen/`.

## Public Resources

| Resource | Link |
| --- | --- |
| Benchmark and prepared data | [sharryXR/asil-benchmark](https://huggingface.co/datasets/sharryXR/asil-benchmark) |
| Singularity images | [sharryXR/asil-benchmark-images](https://huggingface.co/datasets/sharryXR/asil-benchmark-images) |
| Qwen3.5-2B SFT | [sharryXR/asil-qwen35-2b-sft](https://huggingface.co/sharryXR/asil-qwen35-2b-sft) |
| Qwen3.5-2B RL | [sharryXR/asil-qwen35-2b-rl](https://huggingface.co/sharryXR/asil-qwen35-2b-rl) |
| Qwen3.5-9B SFT | [sharryXR/asil-qwen35-9b-sft](https://huggingface.co/sharryXR/asil-qwen35-9b-sft) |
| Qwen3.5-9B RL | [sharryXR/asil-qwen35-9b-rl](https://huggingface.co/sharryXR/asil-qwen35-9b-rl) |
| Model collection | [ASIL models](https://huggingface.co/collections/sharryXR/asil-models-6a1e9faf39fe6ce4eb4626e1) |

Per-task run artifacts such as screenshots, trajectories, and score files are
not part of the public release.

## License And Contact

Code is released under Apache-2.0. Data and documentation are released under
CC BY 4.0 unless an individual repository card states otherwise. Model
checkpoints inherit the applicable base-model terms.

Questions: Rui Xie (`sharryXR@sjtu.edu.cn`) and Lu Chen
(`chenlusz@sjtu.edu.cn`). Citation metadata is provided in `CITATION.cff`.
