# ASIL: Replacing Screenshot-and-Click with Structured State and Semantic Actions

<p align="center">
  <strong>Agent-Software Interaction Layer</strong><br>
  Structured state observations and code-executable semantic actions for
  software-operating agents.
</p>

<p align="center">
  <a href="docs/ASIL_EMNLP_2026_Findings.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper PDF"></a>
  <a href="https://huggingface.co/collections/sharryXR/asil-models-6a1e9faf39fe6ce4eb4626e1"><img src="https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Data-FFD21E?logo=huggingface&amp;logoColor=000" alt="Hugging Face models and data"></a>
  <a href="https://huggingface.co/datasets/sharryXR/asil-benchmark"><img src="https://img.shields.io/badge/Benchmark-380%20Tasks-2F6FEB" alt="ASIL benchmark"></a>
  <a href="https://github.com/sharryXR/ASIL"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="GitHub code"></a>
  <a href="#license-and-contact"><img src="https://img.shields.io/badge/License-Apache%202.0-2E8B57.svg" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <strong>News:</strong> ASIL has been accepted to
  <strong>Findings of EMNLP 2026</strong>.
</p>

<p align="center">
  <a href="docs/ASIL_EMNLP_2026_Findings.pdf">Paper</a> |
  <a href="#overview">Overview</a> |
  <a href="#headline-results">Results</a> |
  <a href="#install">Install</a> |
  <a href="#benchmark">Benchmark</a> |
  <a href="#assisted-software-onboarding">Software Onboarding</a> |
  <a href="#public-resources">Resources</a> |
  <a href="#citation">Citation</a>
</p>

---

## Overview

Powerful code agents can execute scripts, call tools, and manage files, yet
many important applications remain accessible primarily through graphical user
interfaces. ASIL replaces the screenshot-and-click loop with an agent-native
interface that exposes software through structured JSON observations and
code-executable semantic actions, realized through the deepest feasible access
path for each application.

We instantiate ASIL across 15 applications and a benchmark of 300
single-application and 80 multi-application tasks. The same task definitions,
initial artifacts, and software-state validators are shared across ASIL and GUI
control, isolating the effect of the observation-action interface. Structured
observations and semantic actions also provide compact, verifiable traces for
SFT and evaluator-backed on-policy RL.

This release contains the runtime protocol, 15-application adapters, GUI and
native-interface baselines, independent raw-state validation,
Docker/Singularity environments, and the assisted software onboarding
framework described in the paper. The Paper link currently serves the accepted
camera-ready PDF and will be updated to arXiv after the preprint is released.

## Headline Results

Main 380-task benchmark scores from the accepted paper:

| Model | ASIL | Screenshot GUI | ASIL - GUI |
| --- | ---: | ---: | ---: |
| Kimi K2.5 | **84.8** | 4.9 | +79.9 |
| GPT-5.4 | 81.6 | 8.4 | +73.2 |
| sonnet4.6 | 81.2 | 5.6 | +75.6 |
| Qwen3.6-plus | 81.1 | 2.9 | +78.2 |
| Qwen3.5-27B | 75.8 | 4.7 | +71.1 |

Small-scale training under the ASIL modality:

| Model | Base | SFT | RL |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B | 58.0 | 72.1 | **74.4** |
| Qwen3.5-9B | 66.6 | 80.4 | **82.2** |

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

## Citation

If you find ASIL useful, please cite the accepted paper:

```bibtex
@inproceedings{xie2026asil,
  title={{ASIL}: Replacing Screenshot-and-Click with Structured State and Semantic Actions},
  author={Xie, Rui and Chen, Lu},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026}
}
```

## License And Contact

Code is released under Apache-2.0. Data and documentation are released under
CC BY 4.0 unless an individual repository card states otherwise. Model
checkpoints inherit the applicable base-model terms.

Questions: Rui Xie (`sharryXR@sjtu.edu.cn`) and Lu Chen
(`chenlusz@sjtu.edu.cn`). Citation metadata is provided in `CITATION.cff`.
