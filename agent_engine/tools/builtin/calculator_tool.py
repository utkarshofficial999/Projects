"""
agent_engine.tools.builtin.calculator_tool
==========================================

A safe arithmetic calculator tool.

Evaluates a mathematical expression using Python's ``ast`` module to
parse and evaluate only a whitelist of arithmetic nodes. This avoids
the security risks of ``eval`` while still supporting ``+``, ``-``,
``*``, ``/``, ``//``, ``%``, ``**``, and unary minus.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict, Union

from pydantic import BaseModel, Field

from agent_engine.tools.base_tool import BaseTool, ToolExecutionError
from agent_engine.tools.registry import register_tool

# Whitelist of allowed binary operators
_BINARY_OPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Whitelist of allowed unary operators
_UNARY_OPS: Dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorArgs(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description=(
            "A mathematical expression, e.g. '2 + 3 * 4' or "
            "'(10 / 2) ** 3'. Only arithmetic operators are allowed."
        ),
    )


@register_tool
class CalculatorTool(BaseTool):
    """Safely evaluate an arithmetic expression."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Supports +, -, *, /, //, "
        "%, **, and parentheses. Example: '2 + 3 * 4'."
    )
    arguments_schema = CalculatorArgs

    def execute(self, args: CalculatorArgs) -> str:
        """Evaluate the expression and return the result as a string."""
        try:
            result = self._safe_eval(args.expression)
        except ToolExecutionError:
            raise
        except ZeroDivisionError:
            raise ToolExecutionError(
                self.name, "Division by zero."
            ) from None
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(
                self.name, f"Invalid expression: {exc}"
            ) from exc

        # Format: show as int if the result is a whole number
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)

    # ------------------------------------------------------------------
    # AST-based safe evaluator
    # ------------------------------------------------------------------

    def _safe_eval(self, expression: str) -> Union[int, float]:
        """Parse and evaluate *expression* using a restricted AST walker.

        Only numeric literals and whitelisted arithmetic operators are
        permitted. Any other node type raises ``ToolExecutionError``.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(
                self.name, f"Syntax error: {exc.msg}"
            ) from exc

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node."""
        # Numeric literal (Python 3.8+: ast.Constant)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ToolExecutionError(
                self.name,
                f"Only numeric literals are allowed, got {type(node.value).__name__}",
            )

        # Binary operation: a op b
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BINARY_OPS:
                raise ToolExecutionError(
                    self.name,
                    f"Operator '{op_type.__name__}' is not allowed.",
                )
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _BINARY_OPS[op_type](left, right)

        # Unary operation: +a or -a
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ToolExecutionError(
                    self.name,
                    f"Unary operator '{op_type.__name__}' is not allowed.",
                )
            operand = self._eval_node(node.operand)
            return _UNARY_OPS[op_type](operand)

        raise ToolExecutionError(
            self.name,
            f"Unsupported expression element: {type(node).__name__}",
        )
