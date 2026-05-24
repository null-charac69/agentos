"""
tools/code_tools.py
────────────────────
Sandboxed Python execution tool.

Why RestrictedPython over subprocess/Docker-in-Docker?
  - No Docker socket privilege needed
  - No extra container overhead
  - Sufficient for data-analysis / math snippets an LLM would generate
  - Compiles code to restricted bytecode that blocks dangerous builtins

What IS allowed:
  - math, statistics, json, re, datetime, collections, itertools, functools
  - print() (captured to stdout buffer)
  - Basic data manipulation

What is BLOCKED:
  - open(), os, sys, subprocess, socket, __import__ tricks
  - File I/O, network access, os.system, eval with __builtins__
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import TYPE_CHECKING

import structlog
from RestrictedPython import compile_restricted, safe_globals, safe_add_global
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence

if TYPE_CHECKING:
    from core.state import CodeResult

logger = structlog.get_logger(__name__)

# ─── Allowed imports whitelist ────────────────────────────────────────────────

_ALLOWED_MODULES = {
    "math",
    "statistics",
    "json",
    "re",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "random",
    "decimal",
    "fractions",
    "string",
}


def _guarded_import(name: str, *args, **kwargs):
    """Replacement for __import__ that enforces the allowlist."""
    if name not in _ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed in the sandbox. "
            f"Allowed modules: {sorted(_ALLOWED_MODULES)}"
        )
    return __import__(name, *args, **kwargs)


# ─── Restricted execution globals ─────────────────────────────────────────────

def _build_restricted_globals() -> dict:
    restricted = safe_globals.copy()
    restricted["__builtins__"] = safe_builtins.copy()
    restricted["__builtins__"]["__import__"] = _guarded_import
    restricted["_getiter_"] = iter
    restricted["_getattr_"] = getattr
    restricted["_write_"] = lambda x: x          # allow attribute writes
    restricted["_inplacevar_"] = lambda op, x, y: x  # allow in-place ops
    restricted["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    return restricted


# ─── Executor ────────────────────────────────────────────────────────────────

def execute_code(
    code: str,
    task_id: str,
    timeout: int = 10,
) -> CodeResult:
    """
    Execute a Python snippet inside a RestrictedPython sandbox.

    Args:
        code: The Python source to execute.
        task_id: Identifier of the planner task that requested this execution.
        timeout: Maximum wall-clock seconds before we give up.
                 (Implemented via threading.Timer — not process-level kill.)

    Returns:
        A CodeResult dict with stdout, stderr, success flag, and timing.
    """
    start = time.monotonic()

    # Compile to restricted bytecode
    try:
        byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
    except SyntaxError as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("code_compile_error", task_id=task_id, error=str(exc))
        return {
            "task_id": task_id,
            "code": code,
            "stdout": "",
            "stderr": f"SyntaxError: {exc}",
            "success": False,
            "execution_time_ms": elapsed,
        }

    # Capture output
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    local_vars: dict = {}
    restricted_globals = _build_restricted_globals()

    # Execution with timeout
    import threading

    exec_exception: list[BaseException] = []

    def _run():
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(byte_code, restricted_globals, local_vars)  # noqa: S102
        except Exception as e:
            exec_exception.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    elapsed = (time.monotonic() - start) * 1000

    if thread.is_alive():
        logger.warning("code_execution_timeout", task_id=task_id, timeout=timeout)
        return {
            "task_id": task_id,
            "code": code,
            "stdout": stdout_buf.getvalue(),
            "stderr": f"TimeoutError: execution exceeded {timeout}s",
            "success": False,
            "execution_time_ms": elapsed,
        }

    if exec_exception:
        err_msg = f"{type(exec_exception[0]).__name__}: {exec_exception[0]}"
        logger.warning("code_execution_error", task_id=task_id, error=err_msg)
        return {
            "task_id": task_id,
            "code": code,
            "stdout": stdout_buf.getvalue(),
            "stderr": err_msg,
            "success": False,
            "execution_time_ms": elapsed,
        }

    logger.debug("code_execution_success", task_id=task_id, elapsed_ms=elapsed)
    return {
        "task_id": task_id,
        "code": code,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "success": True,
        "execution_time_ms": elapsed,
    }
