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

---

## Quality Gates

No mandatory quality gates in this sandbox. You may use ruff locally for formatting/linting if helpful.

---

## Typing

### Typing Guidelines (pragmatic, not performative)

* Prefer **precise** types: `Iterable[str]`, `Mapping[str, str]`, `list[str]` (3.9+ syntax).
* Use `|` operator for unions instead of `Union`: `str | None` not `Union[str, None]` (3.10+ syntax).
* Model structured dicts with `TypedDict`.
* Use `Literal` for string enums; `Enum` only if you need methods.
* Avoid leaking `Any`; if unavoidable, **contain** it at the boundary.

### When to use `from __future__ import annotations` (Python 3.13)

In Python 3.13, you **generally do NOT need** this import for most code using modern syntax.

**ONLY add it when you have:**
* **Self-referencing return types** (methods that return the class they're defined in)
* **Forward references** (referencing a class before it's defined)
* **Circular import issues** with type hints

**Example that needs it:**
```python
from __future__ import annotations

class Config(BaseModel):
    def merge(self, other: Config) -> Config:  # Returns same class
        return self.model_copy(update=other.model_dump())
```

**Without the import**, you'd need: `-> "Config"` (quoted string).

**DON'T use it** just for modern syntax like `list[str]`, `dict[str, int]`, or `X | Y` unions - these work fine in Python 3.13 without the import.

## Advanced Typing & Generics

Use modern typing to express intent and make refactors safe.

### Python 3.13+ Modern Syntax

**Always use Python 3.13+ type parameter syntax.** Use `[T]` instead of `TypeVar` + `Generic`.

### Generics

```python
# Modern syntax (Python 3.13+)
class Box[T]:
    def __init__(self, value: T) -> None:
        self.value = value

def first[T](items: list[T]) -> T:
    if not items:
        raise ValueError("empty")
    return items[0]
```

* For bounded types, use `[T: Base]` syntax: `def process[T: BaseModel](item: T) -> T`
* For constrained types, still use `TypeVar(..., str, int)` when needed (rare).

### Variadic generics (3.11+)

```python
from typing import TypeVarTuple, Generic

Ts = TypeVarTuple("Ts")

class Row(Generic[*Ts]):  # a row with N typed columns
    def __init__(self, *cols: *Ts) -> None:
        self.cols = cols
```

### ParamSpec (higher-order functions)

Preserve call signatures in decorators/wrappers.

```python
from typing import Callable, ParamSpec, TypeVar
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def log_calls(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(fn.__name__, args, kwargs)
        return fn(*args, **kwargs)
    return wrapper
```

### Protocols & structural typing

Prefer behavior over inheritance.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SupportsClose(Protocol):
    def close(self) -> None: ...

def shutdown(resource: SupportsClose) -> None:
    resource.close()
```

### TypedDict (structured dicts)

```python
from typing import TypedDict, NotRequired

class UserRow(TypedDict):
    id: int
    email: str
    name: NotRequired[str]
```

### Overloads for precise APIs

```python
from typing import overload

@overload
def get(idx: int) -> int: ...
@overload
def get(idx: str) -> str: ...

def get(idx):  # runtime impl
    return idx
```

### Type narrowing

* Use `isinstance`, `match`, and small guards with `TypeGuard` when needed.

```python
from typing import TypeGuard

def is_int_str(s: str) -> TypeGuard[str]:
    return s.isdigit()
```

### NewType & domain IDs

```python
from typing import NewType
UserId = NewType("UserId", int)
```

### Annotated (metadata)

Use `typing.Annotated` to attach units/constraints (works well with Pydantic v2).

```python
from typing import Annotated
from pydantic import Field

UserName = Annotated[str, Field(min_length=3, max_length=50)]
```

#### Rules

* Prefer `object` at untyped boundaries over `Any`.
* Export stable public types; keep internals private.
* Keep overload sets **small** and tested.

## Docstrings

* Use **Google-style** docstrings for public functions/classes (flake8-docstrings checks).
* Write meaningful descriptions on Pydantic fields.

## Commenting & Docs (Keep it short, explain **why**, not **what**)

### Rules of thumb

* **Prefer self-explanatory code** over comments. Rename > comment.
* **Comment the *why*, not the what.** If you’re explaining syntax or a trivial branch, rewrite the code instead.
* **Two lines max** per inline comment. Use a docstring for public APIs; keep those short and high-signal.
* **No story time.** Avoid verbose LLM-style “explanations.” If the insight doesn’t change a decision, delete it.
* **Don’t duplicate types.** Type hints and names should carry most meaning.
* **No secrets or PII** in comments. Ever.
* **Kill dead code.** Don’t park chunks under comments; If temporarily commented, mark with `# //`

### Conventions

* `# ! ALERT:` dangerous edges (security/data loss/perf). One line max.
* `# ? QUESTION:` for reviewers; resolve or convert to TODO before merge.
* `# TODO:` small, actionable, with owner/issue.
* `# * NOTE:` brief rationale not obvious from code/tests.
* `# //` commented-out code (rare; state why & removal condition).

---

## Tiny Style Nits (consistency wins)

* f-strings over `.format`.
* Constants UPPER_SNAKE at module top; no magic numbers.
* Prefer `return` early; avoid deep nesting.
* Logging over `print`.

---

## Pythonic Practices

* Prefer **small, pure functions**; extract IO from core logic.
* No mutable default args; use `None` and set inside.
* Raise precise exceptions; don’t swallow errors silently.
* Keep functions to a single level of abstraction.
* Prefer **dataclasses** for simple data containers; Pydantic **BaseModel** for API I/O and validation.
* Use **type hints** everywhere (PEP 484/561).
* Prefer **explicit over implicit**; keep functions short; extract helpers.
* Use **pathlib** over `os.path`.

## More Pythonic Best Practices

### Idioms that read well

* Prefer clear **guard clauses** over deep nesting.
* Use **EAFP** (“try first, handle errors”) at boundaries; use **LBYL** (“check first”) only when it avoids expensive/unsafe work.
* Keep **comprehensions** simple; if there’s an `if/else` and a function call, consider a loop.
* Prefer **truthiness** checks: `if items:` instead of `if len(items) > 0:`.

```python
# good
names = [u.name for u in users if u.active]

# stream, don’t build whole lists if you only count
active_count = sum(1 for u in users if u.active)
```

### Iteration & collections

* Use `enumerate(seq)` instead of manual indices.
* Use `zip(a, b, strict=True)` (3.10+) to catch length mismatches.
* Prefer `any()` / `all()` for predicates; `min`/`max` with `default=...`.
* Reach for the stdlib power tools:

  * `collections.Counter/defaultdict/deque(maxlen=...)`
  * `itertools.pairwise`, `groupby` (pre-sort!), `accumulate`
  * `heapq.nlargest` for top-k; `bisect` for sorted inserts/searches

### Files, paths & text

* Use **`pathlib.Path`** and its helpers: `Path.read_text()` / `write_text(encoding="utf-8")`.
* For multiline strings: `textwrap.dedent` to keep indentation sane.

### Functional helpers

* Cache pure results with `functools.lru_cache(maxsize=...)`.
* Prefer generators to stream large data; return `Iterator[T]`/`Iterable[T]` not lists when possible.

### Imports & module hygiene

* Absolute imports; group stdlib / third-party / local with blank lines.
* Put executable code under `if __name__ == "__main__":` and call a `main()`.

### Subprocess & shell safety

* Use `subprocess.run([...], check=True, capture_output=True, text=True)`; avoid shell=True unless necessary.
* Use `shlex.split` / `shlex.quote` to handle user input safely.

### Pattern matching (3.10+)

* Use `match/case` when it **simplifies** branching (parsing tokens, state machines). Keep patterns small; add a catch-all.

```python
match kind, payload:
    case ("user", {"id": int(id)}):
        ...
    case _:
        raise ValueError("unsupported")
```

### Micro-optimizations

* Hoist repeated attribute lookups inside tight loops.
* Prefer tuple literals for immutable constants.
* Avoid excessive object creation in hot paths (consider `slots`/`__slots__`).

---

## Pythonic Code Patterns (comprehensions, itertools, generators, decorators, OOP)

Your code should be pythonic - adhere to these principles:

### Comprehensions & expressions

* Prefer **list/set/dict comprehensions** for simple transforms & filters.
* Keep them single-purpose; if nested or complex, switch to a loop.

```python
squares = {n: n*n for n in range(10) if n % 2 == 0}
unique_names = {u.name for u in users if u.active}
```

### itertools you should know (3.12+)

* `itertools.batched(iterable, n)` for chunking streams.
* `itertools.accumulate`, `pairwise`, `groupby` (sort by the same key!), `islice`, `chain.from_iterable`.

```python
from itertools import batched, groupby
for batch in batched(records, 500):
    process(batch)

records.sort(key=lambda r: r.country)
for country, rows in groupby(records, key=lambda r: r.country):
    ...
```

### Generators & iterators

* Use `yield` to stream data; return `Iterator[T]` when callers can consume lazily.
* Compose with `yield from` to keep functions small.

```python
from collections.abc import Iterator

def read_lines(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
```

### Decorators (typed & safe)

* Always use `functools.wraps`.
* Type with `ParamSpec` + `TypeVar` to preserve signatures.

```python
from typing import Callable, ParamSpec, TypeVar
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def retry(times: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def outer(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def inner(*args: P.args, **kw: P.kwargs) -> R:
            for attempt in range(times):
                try:
                    return fn(*args, **kw)
                except TransientError:
                    if attempt == times - 1:
                        raise
        return inner
    return outer
```

### Context managers

* Prefer `with` to ensure cleanup; use `contextlib.contextmanager` for lightweight contexts; `ExitStack` for dynamic resources.

### OOP done the Pythonic way

* Default to **dataclasses** (`frozen=True` for value objects, `slots=True` for many instances).
* Prefer **composition** over inheritance; use **Protocols** to encode behavior.
* Keep methods small; one responsibility per class.

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Money:
    amount: int  # cents
    currency: str

class PriceFormatter:
    def format(self, m: Money) -> str:
        return f"{m.amount/100:.2f} {m.currency}"
```

---

## Async & Concurrency

* Don't block the event loop in `async` code; use `asyncio.to_thread` for blocking I/O.
* Add **timeouts** and **retries with jitter** for external calls.
* Keep background tasks **idempotent** and **observable** (logs/metrics).
* Use `httpx.AsyncClient` as a **context manager**; one client per scope.
* Limit parallelism with `asyncio.Semaphore`; cancel on shutdown.
* Prefer `asyncio.TaskGroup` (3.11+) for structured concurrency.
* Always handle cancellation properly with `try/except asyncio.CancelledError`.

---

## FastAPI-specific Patterns (if used)

* Model requests/responses with Pydantic; annotate fields with `description`.
* Keep routes thin; put business logic in services/helpers.
* Generate and publish OpenAPI when routes or schemas change.
* Return declared models; avoid raw dicts unless trivial.
* Enforce auth/permissions at the router or dependency layer.

### App Factory

* Keep app creation in `create_app()` with ENV-driven behavior (dev/test show docs; prod hides them).
* All middlewares are registered centrally (`errors.py`, `middlewares.py`).

### Routes

* Use `routers` in `main.py` (or a new module) and include via `app.include_router`.
* Tag routes meaningfully.
* **Security**: Ensure auth is injected in prod via dependencies. Don’t bypass in prod paths. Ask user how to implement this if needed.

### Requests/Responses

* Define schemas in `src/pkg/schemas.py` or a domain-specific module.
* Use `Field(..., description="...")` for OpenAPI clarity.
* Be **strict** about response models; avoid `dict` returns unless trivial (health).
* Use **BackgroundTasks** for fire-and-forget jobs, but keep them idempotent and small.

---

## Configuration & Secrets

* Use `pydantic-settings` or env vars; never hard-code secrets.
* Provide `.env.example` only; don’t commit real `.env`.
* Use `dotenv` to load `.env` (`load_dotenv(find_dotenv())`).

---

## Logging

* Use `logger.info/debug/warning/error`.
* Prefer structured logging: `logger.info("Message: {id}", id=thing_id)`.
* Keep PII and secrets out of logs.
* Log at the edges (requests/responses, retries, failures), not on every line.

---

## Testing (simple)

- Use pytest for small unit tests of core pure logic when helpful.
- No coverage thresholds, no TDD requirement, no integration/e2e tests, no CI gates.

How to run:

```bash
uv run pytest -q
```

---

## Error Handling

* Validate inputs early (Pydantic or manual checks).
* Use domain-specific exceptions for expected errors. No blanket `except Exception` unless at process boundary.
* Map exceptions to clean error responses if building an API.
* For unexpected errors:
  * log with stacktrace
  * return a generic message to callers
  * avoid leaking internals
* Use `try`/`except` blocks to handle errors gracefully.
* Always clean up resources (e.g., close files, release locks) in `finally` blocks.
* When caching an error, log it with stacktrace context for debugging (`logger.opt(exception=e).error(...)`).
* Always attach timeouts to I/O; prefer retry **only** for transient errors.

---

## Performance & Memory

* Don’t optimize prematurely. Measure first.
* Avoid N+1 I/O; batch where possible.
* For hot paths:
  * prefer locals over repeated attribute lookups,
  * `dataclass(slots=True)` for many instances,
  * consider `lru_cache` for pure functions (with explicit `maxsize`).

---

## Packaging & Versioning

* Maintain `pyproject.toml` as the single source of truth.
* Use semantic versioning; keep a `CHANGELOG`.

---

## CLI & UX (if building tools)

* Prefer `typer` (typed, ergonomic) or `argparse` for zero deps.
* Provide `--version`, `--verbose`, `--help`. Exit codes matter.

---

## Docker & Deployment

* Use uv for fast, reproducible installs.
* Prefer slim base images, multi-stage builds, non-root user, minimal runtime tools.
* Cache uv artifacts to speed builds; add a sane `.dockerignore`.

### Minimal example

```dockerfile
FROM python:3.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
COPY src/ src/
USER 65532:65532
CMD ["uv", "run", "python", "-m", "yourpkg"]
```

---

## Snippets

### Typed Function Template

```python
from typing import Iterable

def select_top_k(items: Iterable[str], k: int) -> list[str]:
    """Return top-k items by some metric.

    Args:
        items: Items to select from.
        k: Number to keep; must be >= 0.

    Returns:
        The top-k items.

    Raises:
        ValueError: If k is negative.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    ranked = sorted(items)
    return ranked[:k]
```

### Generic Function Template (Python 3.13+)

```python
def wrap[T](value: T) -> dict[str, T]:
    """Wrap a value in a dict.

    Args:
        value: Value to wrap.

    Returns:
        Dict with the wrapped value.
    """
    return {"data": value}
```

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


