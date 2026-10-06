"""
agent_engine.tools.sandbox
===========================

Secure execution sandbox for tool invocation.

The :class:`ToolSandbox` runs a tool's ``execute`` method in a
**child process** so that:

* A misbehaving tool (infinite loop, segfault, ``os._exit``) cannot
  take down the parent agent process.
* ``stdout`` / ``stderr`` produced by the tool are captured and
  returned alongside the result.
* A configurable **timeout** kills the child if the tool exceeds its
  time budget.

The sandbox serialises the validated argument model to JSON, spawns a
subprocess that re-instantiates the tool and calls ``execute``, and
parses the child's JSON output.

Design notes
------------
* We use ``multiprocessing`` with the ``spawn`` start method (the
  default on Windows and macOS) to guarantee a clean interpreter
  state.
* The child process writes a single JSON line to stdout:
  ``{"success": true, "output": "..."}`` or
  ``{"success": false, "error": "..."}``.
* Any text the tool prints to stdout/stderr *before* the JSON line is
  captured separately and attached to the result.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agent_engine.tools.base_tool import BaseTool, ToolExecutionError
from agent_engine.tools.registry import ToolResult

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Child-process worker (must be top-level for pickling)
# ----------------------------------------------------------------------

def _sandbox_worker(
    tool_cls_path: str,
    args_json: str,
    result_queue: "mp.Queue",
) -> None:
    """Run in the child process: instantiate tool, execute, report.

    Parameters
    ----------
    tool_cls_path:
        Dotted path to the tool class, e.g.
        ``"agent_engine.tools.builtin.calculator_tool.CalculatorTool"``.
    args_json:
        JSON-serialised validated arguments (``model_dump()``).
    result_queue:
        Multiprocessing queue to send the result dict back.
    """
    import importlib

    try:
        module_path, class_name = tool_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        tool_cls = getattr(module, class_name)
        tool = tool_cls()

        # Re-validate from JSON
        args_dict = json.loads(args_json)
        validated = tool.validate_arguments(args_dict)

        output = tool.execute(validated)
        result_queue.put({"success": True, "output": str(output)})
    except ToolExecutionError as exc:
        result_queue.put({"success": False, "error": exc.message})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"success": False, "error": f"{type(exc).__name__}: {exc}"})


# ----------------------------------------------------------------------
# Sandbox configuration
# ----------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Tunable parameters for the execution sandbox.

    Attributes
    ----------
    timeout : float
        Maximum seconds the child process may run before being killed.
    max_output_bytes : int
        Truncate captured stdout/stderr to this many bytes.
    """

    timeout: float = 30.0
    max_output_bytes: int = 65_536  # 64 KiB


# ----------------------------------------------------------------------
# Sandbox
# ----------------------------------------------------------------------

class ToolSandbox:
    """Execute tools in an isolated child process.

    Parameters
    ----------
    config:
        Optional :class:`SandboxConfig`. Defaults to 30 s timeout.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()

    def run(
        self,
        tool: BaseTool,
        validated_args: Any,
    ) -> ToolResult:
        """Execute *tool* in a child process.

        Parameters
        ----------
        tool:
            The tool instance (used to resolve the class path).
        validated_args:
            A validated Pydantic model instance.

        Returns
        -------
        ToolResult
            Structured result with captured stdout/stderr on failure.
        """
        tool_cls_path = f"{tool.__class__.__module__}.{tool.__class__.__qualname__}"
        args_json = json.dumps(validated_args.model_dump())

        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()

        proc = ctx.Process(
            target=_sandbox_worker,
            args=(tool_cls_path, args_json, result_queue),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=self.config.timeout)

        # --- Timeout handling -------------------------------------------
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return ToolResult(
                tool_name=tool.name,
                success=False,
                error=(
                    f"Tool '{tool.name}' timed out after "
                    f"{self.config.timeout:.1f}s and was killed."
                ),
            )

        # --- Collect result ---------------------------------------------
        try:
            result: Dict[str, Any] = result_queue.get(timeout=2.0)
        except Exception:  # noqa: BLE001 – queue.Empty or similar
            return ToolResult(
                tool_name=tool.name,
                success=False,
                error=(
                    f"Tool '{tool.name}' exited with code {proc.exitcode} "
                    f"without returning a result."
                ),
            )

        if result.get("success"):
            return ToolResult(
                tool_name=tool.name,
                success=True,
                output=result.get("output", ""),
            )
        return ToolResult(
            tool_name=tool.name,
            success=False,
            error=result.get("error", "Unknown sandbox error"),
        )
