# Singularity Backend

ASIL includes Singularity definitions for the eval runtime, Gitea, OBS mock,
code-server, JupyterLab, and draw.io.

Build images on a host with Singularity 4.x:

```bash
bash scripts/build_singularity_images.sh \
  --sif-dir singularity/images \
  --force
```

For hosts that require local sudo builds:

```bash
bash scripts/build_singularity_images.sh \
  --sif-dir singularity/images \
  --force \
  --sudo-build \
  --no-module-load
```

Run a managed evaluation with one isolated service stack per worker:

```bash
PYTHONPATH=src python scripts/run_evaluation_singularity_managed.py \
  --num-envs 8 \
  --sif-dir singularity/images \
  --task-set test_full15.json \
  --participant asil \
  --asil-execution deterministic \
  --output-dir results/singularity \
  --output results/singularity/results.json
```

SIF binaries are not stored in Git. The released images are available from
[sharryXR/asil-benchmark-images](https://huggingface.co/datasets/sharryXR/asil-benchmark-images).
