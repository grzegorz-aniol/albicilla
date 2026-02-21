# Agent Sandbox

Project's tech stack:
* Python 3.13+
* CLI based application
* uv as package management system

---

Appended rules from python-template:

# AGENT.md — Coding Agent Playbook (Python)

A concise, general playbook for an autonomous coding agent writing **modern Python**. It's framework-agnostic (works great with FastAPI, CLIs, data tools, etc.) and emphasizes **uv** for dependency management and **make** for tasks.

---

### Agent interaction

* As an agent, interacting with the user, you should follow this principles:
  * You need to minimize number of external files printed in your responses as this increase LLM token usage. Refer to the file name instead. 

---

## Mission

Deliver small, readable Python changes with clean APIs and pragmatic typing. Prefer clarity over cleverness.

---

## Tooling Overview

* **Python:** 3.13+ (use `.python-version` to pin locally)
* **Package manager:** **uv**
* **Task runner:** **make**
* **Code quality (optional):** `ruff` (lint & format); `mypy` if helpful
* **Testing:** `pytest`
* **Config:** `pydantic-settings` and `dotenv`
* **Logging:** `loguru` with structured messages
* **HTTP (async):** `httpx` (avoid blocking calls in async code)

---
## Documentation / Contracts / Rules

Project keep small set of documents in `docs/` folder.
Review and use whenever you plan any new implementation or want to understand requirements, contracts or conventions. 
Important: If you change such then update the relevant docs and contracts - but keep docs very concise.

---

## Agent Workflow (Always)

1. **Plan** Before coding, list the files you’ll touch, data models you’ll add/modify, and any tests you might write if helpful.
2. **Type everything.** All functions, variables where helpful, Pydantic models, etc. Avoid `Any` unless unavoidable; prefer precise types and `TypedDict` when useful.
3. **Keep contracts and documentation up to date.** If you change anything, update:
   * Tests
   * README/docs (if applicable)
4. Write simple unit tests with pytest for non-trivial pure logic when helpful. No TDD requirement.
5. If tests exist, run them with:

```bash
uv run pytest -q
```

---

## Project Layout (Recommended)

```text
project/
├─ pyproject.toml
├─ uv.lock
├─ .pre-commit-config.yaml
├─ .python-version
├─ README.md
├─ src/yourpkg/...
└─ tests/
    ├─ unit/
    └─ integration/
```

> Use `src/` layout to avoid accidental imports from the working dir.

---

## Dependencies with uv

* **Install all groups**:

```bash
make install  # or: uv sync --group dev --group test --group qa
```

Assume `uv` is used for all commands. Assume that environment is done for you upfront so you do not need to activate it manually.

* **Add a runtime dep**:

```bash
uv add httpx
```

* **Add a dev-only dep**:

```bash
uv add --group dev ruff
```

## Running Python Scripts with uv

Always run Python scripts with the `uv run` command:

```bash
uv run python path/to/script.py
```

---

## Tasks with Make

All tasks are defined in the `Makefile`. Run them with:

```bash
make <task>
```

Available tasks you will commonly use:

* `install` - Install dev/test/qa deps
* `lint` - Run pre-commit hooks (format/lint/types/security)
* `test` - Run quick tests (exclude slow tests)
* `build` - Build sdist+wheel
* `run-local` - Start FastAPI locally with reload
* `run-docker` - Compose up (build)
* `clean` - Clean build/test artifacts

Check available tasks:

```bash
make help  # See all Make targets
```


## Pythonic Code Patterns (comprehensions, itertools, generators, decorators, OOP)

IMPORTANT: Use python-developer skill for any development/planning task.

### Pydantic Model

```python
from pydantic import BaseModel, Field

class FooRequest(BaseModel):
    id: str = Field(..., description="Unique identifier for Foo")
    limit: int = Field(10, ge=0, le=100, description="Max items to return")
```

### Loguru Usage

```python
from loguru import logger

logger.info("Processing foo: {id}", id=foo_id)
logger.debug("Payload: {payload}", payload=payload)  # avoid secrets
```

---

## What to Do When Unsure

* Stop. Write a 3–5 line plan for the change (files, models, tests).
* Ask for confirmation if the plan impacts public APIs or dependencies.
* Default to the safest implementation that passes tests and gates.

---

### Quick Commands Recap

```bash
make help            # see all available commands
make install         # setup
make lint            # format/lint/types/security
make test            # run tests
make run-local       # start dev server
make run-docker      # run in Docker
make build           # build package
make clean           # clean artifacts
```

Optional: If you have unit tests, run `uv run pytest -q`.

That's it. Keep it typed and tidy.

## MCP servers

For get-library-docs you have to specify at least two parameters: context7CompatibleLibraryID and topic. 
Here is the list of well know context7 IDs for known libraries:
* /openai/openai-agents-python

## Coding Assistant rules

# Cline

When the task is complete, never finish on command line that opens the generated/modified file (e.g. open ....).


