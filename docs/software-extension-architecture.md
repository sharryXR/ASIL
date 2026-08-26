# ASIL Software Extension Architecture

ASIL separates application integration from benchmark definition. An adapter
exposes structured state and semantic actions; task definitions and evaluators
specify what a benchmark run asks for and how success is measured.

## Runtime Layers

1. `src/asil/protocol.py` defines observations, elements, actions, environment
   metadata, and serialization.
2. `src/asil/adapter.py` defines the adapter lifecycle: setup, observation,
   action validation, execution, cloning, and context.
3. `src/asil/adapters/` implements application-specific state and action
   bridges.
4. `src/asil/action_schemas/` publishes the semantic action contract shown to
   agents.
5. `src/asil/rendering.py` and adapter render methods expose visible state for
   audit and optional hybrid observation.
6. `src/asil/eval/` loads tasks, evaluates path checkpoints, validates raw final
   state, computes metrics, and writes local run outputs.
7. `src/asil/benchmark.py` provides deterministic, agentic, CLI, and screenshot
   GUI participant entrypoints.

The canonical task source is
`evaluation_examples/examples/<software>/<task_id>.json`. Task-set indexes
select the active domains and task IDs. The evaluator reads the same path-based
checkpoint contract for all participants.

## Integration Patterns

The 15 paper applications use three recurring access-path families:

| Family | Applications | Typical state/action path |
| --- | --- | --- |
| File-backed | Inkscape, LibreOffice Calc/Writer/Impress, draw.io, JupyterLab | Open document formats plus reviewed mutation |
| Native script | Blender, GIMP, Kdenlive, Audacity | Application-native scripting or project graph |
| Service/API | OBS, Gitea, code-server, Thunderbird, Nautilus | WebSocket, REST, filesystem, or desktop service state |

The deepest stable open interface is preferred. Rendering remains a separate
channel because a structurally correct state can still require visible-layout
inspection. Opaque or primarily perceptual behavior therefore needs manual or
hybrid integration.

## Assisted Onboarding

`src/asil/softwaregen/` implements the semi-automated onboarding workflow.

### 1. Evidence Profile

An onboarding profile records software identity, cited evidence for each
observation and action path, allowed hosts and executables, filesystem scope,
known limitations, and required human review.

Qualification returns one of three practical outcomes:

- `direct_declarative`: the generic JSON-file, structured-command, or HTTP
  runtime can use the profile directly;
- `bridge_assisted`: a reviewed parser or native-script bridge is required;
- `out_of_scope`: no evidenced stable read/action path is available.

### 2. Candidate Generation

One typed model call proposes an interface plan. Deterministic assembly then
writes the extension bundle, action schema, adapter wrapper, and generation
report. The provider cannot expand runtime host or executable permissions.

```bash
PYTHONPATH=src python -m asil.softwaregen generate \
  examples/softwaregen/gitea_profile.json \
  --provider openai \
  --model gpt-5.4 \
  --output results/softwaregen/gitea
```

### 3. Fail-Closed Audit

The audit verifies evidence references, JSON pointers, parameter templates,
transport restrictions, allowed hosts/executables, and operation contracts.

```bash
PYTHONPATH=src python -m asil.softwaregen audit \
  results/softwaregen/gitea/extension.json
```

### 4. Host And Docker Probes

Observation-only probes are the default. An action probe is permitted only
when both an explicit action file and `--allow-actions` are provided. Docker
probes mount the reviewed bundle read-only and invoke the same runtime.

```bash
PYTHONPATH=src python -m asil.softwaregen probe \
  results/softwaregen/gitea/extension.json
```

### 5. Evidence Report

`evidence-report` recomputes artifact and bundle hashes, reruns static audit,
checks evidence-reference coverage and provenance, and requires every supplied
action report to contain a validated state change.

The public Gitea example validates `2 -> 3 -> 2` repository restoration on both
host and Docker. Its sanitized measurement and deployment report are in
`examples/softwaregen/`.

## Human Review Boundary

Assisted onboarding does not automate interface discovery, parser or native
bridge implementation, task/evaluator design, GUI synchronization and
rendering, unsupported application semantics, or final benchmark acceptance.

The reference catalog describes recurring access paths across 15 implemented
applications. Those adapters predate the current generator, so the catalog is
not evidence that they were retrospectively generated.

## Adding An Application

1. Gather authoritative interface evidence and write an onboarding profile.
2. Run `qualify`; stop on `out_of_scope`.
3. Generate or deterministically assemble a candidate bundle.
4. Run static audit and review every generated file.
5. Run observation-only probes on host and Docker.
6. Run isolated create/update/delete actions with restoration when appropriate.
7. Implement required parsing, rendering, synchronization, and adapter code.
8. Add action schemas, tasks, evaluator paths, and focused tests.
9. Run the public release validator and the complete test suite.

Generated output is a reviewable candidate, not an automatically trusted
application integration.
