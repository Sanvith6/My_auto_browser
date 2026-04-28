"""
workflow.engine — Step-graph workflow runner.

Features:
  - {{ context.inputs.key }} template resolution for workflow inputs
  - {{ context.steps.step_id.key }} template chaining between steps
  - Retry with exponential backoff
  - Dependency resolution (steps wait for deps)
  - Disk persistence under /data/workflows/
  - Async execution

Context contract (workflow scope):

context:
  scope: workflow
  structure:
    inputs: {}
    steps: {}
    outputs: {}
  lifecycle:
    init: merge(initial_context -> context.inputs)
    per_step: write result to context.steps.<step_id>
    finalize: optional caller-mapped outputs -> context.outputs
  write_policy:
    inputs.*: workflow runner (initial_context)
    steps.*: workflow runner (per-step results)
    outputs.*: workflow caller (optional)
  conflict_resolution:
    steps.*: isolated per-step namespaces (no shared keys)
    inputs.*: last-write-wins on init
    outputs.*: overwrite on finalize
  isolation:
    step_namespace: true
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_WORKFLOWS_ROOT = Path("/data/workflows")
_CONTEXT_KEYS = ("inputs", "steps", "outputs")


# ---------------------------------------------------------------------------
# Step / Workflow definitions
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    id: str
    action: str                         # e.g. "social.youtube.upload" or "research.viral"
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retry_max: int = 2
    retry_backoff_seconds: float = 3.0
    timeout_seconds: float = 120.0
    condition: str = ""                 # Python-like expression (not eval'd; reserved for future)


@dataclass
class WorkflowRun:
    workflow_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: StepStatus = StepStatus.PENDING
    steps: list[WorkflowStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    step_statuses: dict[str, StepStatus] = field(default_factory=dict)
    step_results: dict[str, Any] = field(default_factory=dict)
    step_errors: dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""


ActionFn = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
# (action_name, params_with_resolved_templates, context) → result_dict


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\{\{\s*context\.(\w+(?:\.\w+)*)\s*\}\}")
_MISSING = object()


def _resolve_path(root: Any, key_path: list[str]) -> Any:
    v = root
    for k in key_path:
        if isinstance(v, dict) and k in v:
            v = v[k]
        else:
            return _MISSING
    return v


def _resolve_context_value(context: dict[str, Any], key_path: list[str]) -> Any:
    value = _resolve_path(context, key_path)
    if value is not _MISSING:
        return value
    if not key_path or key_path[0] in _CONTEXT_KEYS:
        return _MISSING
    steps = context.get("steps")
    if isinstance(steps, dict):
        value = _resolve_path(steps, key_path)
        if value is not _MISSING:
            return value
    inputs = context.get("inputs")
    if isinstance(inputs, dict):
        return _resolve_path(inputs, key_path)
    return _MISSING


def _resolve_templates(value: Any, context: dict[str, Any]) -> Any:
    """Recursively replace {{ context.key }} in strings."""
    if isinstance(value, str):
        def replacer(m: re.Match) -> str:
            key_path = m.group(1).split(".")
            v = _resolve_context_value(context, key_path)
            if v is _MISSING:
                return ""
            return str(v)
        return _TEMPLATE_RE.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_templates(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_templates(item, context) for item in value]
    return value


def _normalize_context(initial_context: Optional[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {"inputs": {}, "steps": {}, "outputs": {}}
    if not initial_context:
        return context
    if any(key in initial_context for key in _CONTEXT_KEYS):
        inputs = initial_context.get("inputs", {})
        steps = initial_context.get("steps", {})
        outputs = initial_context.get("outputs", {})
        if isinstance(inputs, dict):
            context["inputs"] = dict(inputs)
        if isinstance(steps, dict):
            context["steps"] = dict(steps)
        if isinstance(outputs, dict):
            context["outputs"] = dict(outputs)
        extra = {k: v for k, v in initial_context.items() if k not in _CONTEXT_KEYS}
        if extra:
            context["inputs"].update(extra)
            logger.warning(
                "workflow.context: merged extra initial_context keys into inputs: %s",
                sorted(extra),
            )
        return context
    context["inputs"] = dict(initial_context)
    return context


# ---------------------------------------------------------------------------
# Workflow engine
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """
    Executes workflow runs against registered action handlers.

    Usage::

        engine = WorkflowEngine()
        engine.register_action("social.youtube.upload", youtube_upload_fn)
        run = await engine.run(workflow_id="social_empire", steps=[...], initial_context={})
    """

    def __init__(self, workflows_root: Path = _DEFAULT_WORKFLOWS_ROOT) -> None:
        self._root = workflows_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._actions: dict[str, ActionFn] = {}

    def register_action(self, name: str, fn: ActionFn) -> None:
        self._actions[name] = fn
        logger.debug("workflow.engine: registered action %r", name)

    async def run(
        self,
        workflow_id: str,
        steps: list[dict[str, Any]],
        initial_context: Optional[dict[str, Any]] = None,
    ) -> WorkflowRun:
        """Execute a workflow, returning the final WorkflowRun."""
        wf_steps = [WorkflowStep(**s) for s in steps]
        run = WorkflowRun(
            workflow_id=workflow_id,
            steps=wf_steps,
            context=_normalize_context(initial_context),
            step_statuses={s.id: StepStatus.PENDING for s in wf_steps},
        )
        run.started_at = time.time()
        run.status = StepStatus.RUNNING
        self._save(run)

        try:
            await self._execute(run)
            run.status = StepStatus.COMPLETED
        except Exception as exc:
            run.status = StepStatus.FAILED
            run.error = "workflow_run_failed"
            logger.error("workflow.engine: run %s failed: %s", run.run_id, exc)
        finally:
            run.finished_at = time.time()
            self._save(run)

        return run

    async def _execute(self, run: WorkflowRun) -> None:
        """Execute steps in dependency order (sequential, dep-aware)."""
        done: set[str] = set()
        failed: set[str] = set()

        while len(done) + len(failed) < len(run.steps):
            progress = False
            for step in run.steps:
                if step.id in done or step.id in failed:
                    continue
                # Skip if any dep failed
                if any(dep in failed for dep in step.depends_on):
                    failed.add(step.id)
                    run.step_statuses[step.id] = StepStatus.FAILED
                    run.step_errors[step.id] = "Dependency failed"
                    progress = True
                    continue
                # Run if all deps done
                if all(dep in done for dep in step.depends_on):
                    try:
                        await self._run_step(step, run)
                        done.add(step.id)
                    except Exception as exc:
                        failed.add(step.id)
                        run.step_statuses[step.id] = StepStatus.FAILED
                        run.step_errors[step.id] = "workflow_step_failed"
                        logger.warning("workflow.step %r failed: %s", step.id, exc)
                    progress = True
                    break  # restart loop to re-evaluate ready steps

            if not progress:
                # No step could make progress — deadlock
                pending = [s.id for s in run.steps if s.id not in done and s.id not in failed]
                for sid in pending:
                    failed.add(sid)
                    run.step_statuses[sid] = StepStatus.FAILED
                    run.step_errors[sid] = "Deadlock: unresolvable dependencies"
                break

        if failed:
            raise RuntimeError(
                f"Workflow {run.workflow_id!r}: steps failed: {sorted(failed)}"
            )


    async def _run_step(self, step: WorkflowStep, run: WorkflowRun) -> None:
        run.step_statuses[step.id] = StepStatus.RUNNING
        self._save(run)

        resolved_params = _resolve_templates(step.params, run.context)

        fn = self._actions.get(step.action)
        if fn is None:
            raise RuntimeError(f"No handler registered for action {step.action!r}")

        last_exc: Optional[Exception] = None
        for attempt in range(step.retry_max + 1):
            try:
                result = await asyncio.wait_for(
                    fn(step.action, resolved_params, run.context),
                    timeout=step.timeout_seconds,
                )
                run.step_results[step.id] = result
                run.step_statuses[step.id] = StepStatus.COMPLETED
                # Merge result into context under context.steps.<step_id>
                steps_context = run.context.get("steps")
                if not isinstance(steps_context, dict):
                    steps_context = {}
                    run.context["steps"] = steps_context
                steps_context[step.id] = result
                self._save(run)
                return
            except asyncio.TimeoutError:
                last_exc = TimeoutError(f"Step {step.id!r} timed out after {step.timeout_seconds}s")
                logger.warning(
                    "workflow.step %r attempt %d/%d timed out",
                    step.id, attempt + 1, step.retry_max + 1,
                )
                if attempt < step.retry_max:
                    backoff = step.retry_backoff_seconds * (2 ** attempt)
                    await asyncio.sleep(backoff)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "workflow.step %r attempt %d/%d failed: %s",
                    step.id, attempt + 1, step.retry_max + 1, exc,
                )
                if attempt < step.retry_max:
                    backoff = step.retry_backoff_seconds * (2 ** attempt)
                    await asyncio.sleep(backoff)

        run.step_statuses[step.id] = StepStatus.FAILED
        raise last_exc or RuntimeError(f"Step {step.id!r} failed after {step.retry_max + 1} attempts")

    def _save(self, run: WorkflowRun) -> None:
        path = self._root / f"{run.run_id}.json"
        data = {
            "workflow_id": run.workflow_id,
            "run_id": run.run_id,
            "status": run.status.value,
            "context": run.context,
            "step_statuses": {k: v.value for k, v in run.step_statuses.items()},
            "step_results": run.step_results,
            "step_errors": run.step_errors,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "error": run.error,
        }
        # Atomic write: temp file + rename, so a crash mid-write never
        # leaves a corrupted run JSON that will break list_runs() later.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def list_runs(self, workflow_id: str = "") -> list[dict[str, Any]]:
        runs = []
        for p in sorted(self._root.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
                if workflow_id and data.get("workflow_id") != workflow_id:
                    continue
                runs.append(data)
            except Exception:
                pass
        return runs
