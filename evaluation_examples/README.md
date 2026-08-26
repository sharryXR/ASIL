# ASIL Task Sets

Task definitions use an OSWorld-style split between canonical task JSON files
and task-set indexes.

- Task definitions: `examples/<software>/<task_id>.json`
- Main single-application index: `test_full15.json` (300 tasks)
- Multi-application index: `test_multi_apps_80.json` (80 tasks)
- Combined main index: `test_full15_multi_apps_380.json` (380 tasks)
- Held-out hard index: `test_full15_realwork_hard.json` (80 tasks)
- Easier comparison index: `test_easy60.json` (60 tasks)

`easy60_selection.json` records public selection attributes for the frozen easy
band. Smoke indexes are deterministic subsets used by Docker and local checks.

Each index maps a software ID to task IDs. Every task ID must resolve to a JSON
file under `examples/`; `scripts/validate_public_release.py` checks this
relationship before publication.
