# Docker Quickstart

The source-built bootstrap targets x86_64 Ubuntu 22.04 or 24.04 with Docker
Engine 24+, Docker Compose v2, Internet access, 30 GiB free disk, and 16 GiB
RAM recommended.

Run host and repository preflight only:

```bash
bash scripts/bootstrap_rebuttal_docker.sh --check-only
```

Build and run the complete deterministic verification:

```bash
bash scripts/bootstrap_rebuttal_docker.sh
```

The command creates a secret-free `.env` only when one does not already exist,
builds the local eval and OBS images, starts pinned Gitea, code-server,
JupyterLab, draw.io, and OBS mock services, runs the frozen four-task smoke, and
then removes only resources under its dedicated `asil-bootstrap` Compose
project. No model API key is required unless `--require-openai` is supplied.

The readiness report is written to
`results/setup/docker_bootstrap_report.json`. Logs and generated run artifacts
remain local under `results/setup/` and are ignored by Git.

Additional modes:

```bash
bash scripts/bootstrap_rebuttal_docker.sh --build-only
bash scripts/bootstrap_rebuttal_docker.sh --verify-only
bash scripts/bootstrap_rebuttal_docker.sh --no-cache
```

The entrypoint retains its historical filename for compatibility. Its
repository inventory checks only files included in this public release.
