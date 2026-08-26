# Contributing

Thank you for improving ASIL.

## Development Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c constraints-host.txt -e ".[dev,eval]"
pytest
```

## Pull Requests

- Keep changes scoped to one runtime, adapter, evaluator, or release concern.
- Add or update tests for behavior changes.
- Do not commit API keys, `.env`, checkpoints, container images, generated
  result directories, local absolute paths, or per-task run artifacts.
- Keep task indexes and their referenced files synchronized.
- Run `python scripts/validate_public_release.py .` before opening a pull
  request.

Training and training-task generation implementations are outside this public
repository's scope.
