# Security

Report security issues privately to `sharryXR@sjtu.edu.cn` and
`chenlusz@sjtu.edu.cn`.

Do not publish API keys, access tokens, private endpoints, model-serving
credentials, SSH keys, or internal machine paths in issues or pull requests.
ASIL reads provider credentials from runtime environment variables or an
untracked `.env` file; credentials must never be embedded in task definitions,
generated onboarding bundles, reports, or traces.

Verify downloaded model, dataset, and Singularity artifacts against the hashes
published in their Hugging Face repository cards before use.
