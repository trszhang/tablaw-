"""Sandboxed Python code execution skill for TabClaw.

Security layers
---------------
1. AST inspection — blocks forbidden imports, dunder attribute access, and
   dangerous built-in calls before any code runs.
2. Restricted namespace — __builtins__ replaced with a curated safe dict so
   the default Python builtins (open, exec, eval, __import__, …) are absent.
3. 30-second daemon thread timeout — limits run-away loops.

Intended for a single-user local app.  Not a hardened multi-tenant sandbox.
"""

import ast
import io
import re
import threading
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import math
import json
import collections
import itertools
import datetime
import statistics
import functools
import operator

# ---------------------------------------------------------------------------
# Allow-/block-lists
# ---------------------------------------------------------------------------

_BLOCKED_IMPORTS: set = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "threading", "multiprocessing",
    "pickle", "marshal", "shelve", "tempfile", "glob",
    "http", "urllib", "requests", "ftplib", "smtplib", "telnetlib",
    "builtins", "__builtin__", "pty", "tty", "signal",
    "fcntl", "termios", "resource", "grp", "pwd", "posix", "nt",
    "winreg", "winsound",
}

_ALLOWED_IMPORTS: set = {
    "math", "re", "json", "collections", "itertools", "datetime",
    "statistics", "functools", "operator", "string", "decimal",
    "fractions", "random", "cmath",
    # Data-science libraries pre-loaded in the namespace
    "pandas", "numpy",
}

_BLOCKED_CALLS: set = {
    "open", "exec", "eval", "compile", "__import__", "input",
    "breakpoint", "getattr", "setattr", "delattr",
    "vars", "dir", "globals", "locals",
}

# Dunder attributes that are safe to reference explicitly
_SAFE_DUNDERS: set = {
    "__len__", "__str__", "__repr__", "__iter__", "__next__",
    "__contains__", "__enter__", "__exit__",
}

# ---------------------------------------------------------------------------
# Safe builtins injected into the execution namespace
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "frozenset": frozenset,
    "str": str, "int": int, "float": float, "bool": bool, "bytes": bytes,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "divmod": divmod, "hash": hash,
    "isinstance": isinstance, "issubclass": issubclass, "hasattr": hasattr,
    "type": type, "repr": repr, "format": format,
    "iter": iter, "next": next, "all": all, "any": any,
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    "True": True, "False": False, "None": None,
    "NotImplemented": NotImplemented,
    # Common exceptions
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "StopIteration": StopIteration, "Exception": Exception,
    "RuntimeError": RuntimeError, "ZeroDivisionError": ZeroDivisionError,
    "NameError": NameError, "OverflowError": OverflowError,
}

_EXECUTION_TIMEOUT = 30  # seconds

_SYSTEM_TIME_KEYWORDS = [
    "更新时间",
    "导出时间",
    "创建时间",
    "修改时间",
    "update_time",
    "export_time",
    "create_time",
]

_BUSINESS_TIME_KEYWORDS = [
    "账期",
    "结算期",
    "业务日期",
    "billing_cycle",
    "账单月",
]


def _semantic_tag_for_column(column_name: Any) -> str:
    """Return semantic tag for time-related columns."""
    normalized = str(column_name).strip()
    normalized_lower = normalized.lower()

    if any(keyword in normalized_lower for keyword in _SYSTEM_TIME_KEYWORDS):
        return "[⚠️系统元数据：绝对禁止用于业务时间过滤或聚合]"
    if any(keyword in normalized_lower for keyword in _BUSINESS_TIME_KEYWORDS):
        return "[🎯核心业务时间维度：请优先使用此列进行时间过滤]"
    return ""


def get_dataframe_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Extract lightweight schema facts for safer code generation."""
    total_rows = max(len(df), 1)
    schema: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        samples = [str(v)[:120] for v in non_null.head(3).tolist()]
        schema[str(col)] = {
            "dtype": str(series.dtype),
            "missing_ratio": round(float(series.isna().sum()) / total_rows, 4),
            "object_samples": samples if str(series.dtype) == "object" else [],
            "semantic_tag": _semantic_tag_for_column(col),
        }
    return schema


def build_tables_schema_context(tables: Dict[str, Dict[str, Any]]) -> str:
    """Build prompt-ready schema text for all currently loaded tables."""
    if not tables:
        return "（当前无可用表）"
    lines: List[str] = []
    for tid, table in tables.items():
        df = table["df"]
        table_name = table.get("name", tid)
        lines.append(f"- 表ID=`{tid}` 名称='{table_name}'")
        schema = get_dataframe_schema(df)
        for col, meta in schema.items():
            ratio = meta["missing_ratio"] * 100
            samples = meta["object_samples"]
            sample_text = f", Object样例={samples}" if samples else ""
            semantic_tag = meta.get("semantic_tag", "")
            display_col = f"{col} {semantic_tag}".rstrip()
            lines.append(
                f"  - 列 `{display_col}`: dtype={meta['dtype']}, 缺失率={ratio:.2f}%{sample_text}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AST safety checker
# ---------------------------------------------------------------------------

class _SafetyChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _BLOCKED_IMPORTS or top not in _ALLOWED_IMPORTS:
                self.violations.append(f"Blocked import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = (node.module or "").split(".")[0]
        if top in _BLOCKED_IMPORTS or (top and top not in _ALLOWED_IMPORTS):
            self.violations.append(f"Blocked import: 'from {node.module} import ...'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") and attr.endswith("__") and attr not in _SAFE_DUNDERS:
            self.violations.append(f"Blocked dunder access: '.{attr}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            self.violations.append(f"Blocked call: '{node.func.id}()'")
        self.generic_visit(node)


def _check_safety(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]
    checker = _SafetyChecker()
    checker.visit(tree)
    return checker.violations


# ---------------------------------------------------------------------------
# Skill entry point
# ---------------------------------------------------------------------------

def execute_python(params: Dict, tables: Dict) -> Any:
    """Run sandboxed Python code with access to all loaded table DataFrames."""
    code = params.get("code", "").strip()
    result_name = params.get("result_name", "code_result")
    # Always profile schemas before execution so upstream prompts can rely on
    # the same inspection logic for defensive code generation.
    _ = {tid: get_dataframe_schema(t["df"]) for tid, t in tables.items()}

    if not code:
        return {"text": "No code provided."}

    # -- Safety gate --
    violations = _check_safety(code)
    if violations:
        msg = "⛔ Code blocked by safety check:\n" + "\n".join(
            f"  • {v}" for v in violations
        )
        return {"text": msg}

    # -- Build print capture --
    output_lines: List[str] = []

    def _safe_print(*args, sep=" ", end="\n", **kwargs):
        if len(output_lines) < 200:  # guard against huge output
            output_lines.append(sep.join(str(a) for a in args) + end)

    # Controlled __import__ that only allows whitelisted modules
    _PRELOADED = {
        "pandas": pd, "pd": pd,
        "numpy": np, "np": np,
        "math": math, "re": re, "json": json,
        "collections": collections, "itertools": itertools,
        "datetime": datetime, "statistics": statistics,
        "functools": functools, "operator": operator,
    }

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top in _PRELOADED:
            return _PRELOADED[top]
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")

    # -- Build execution namespace --
    namespace: Dict[str, Any] = {
        "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import},
        # Data-science libraries (already imported at module level)
        "pd": pd, "pandas": pd,
        "np": np, "numpy": np,
        "math": math, "re": re, "json": json,
        "collections": collections, "itertools": itertools,
        "datetime": datetime, "statistics": statistics,
        "functools": functools, "operator": operator,
        "print": _safe_print,
    }

    # Inject tables by ID and by sanitised name (no-overwrite on collision)
    seen_names: Dict[str, int] = {}
    for tid, t in tables.items():
        df_copy = t["df"].copy()
        namespace[tid] = df_copy
        safe = re.sub(r"\W+", "_", t.get("name", tid)).strip("_")
        if safe:
            count = seen_names.get(safe, 0)
            var = safe if count == 0 else f"{safe}_{count}"
            seen_names[safe] = count + 1
            namespace.setdefault(var, df_copy)

    # Also inject a `tables` mapping {table_id: {"name": ..., "df": DataFrame}}
    # so skill code can iterate over all tables generically
    namespace["tables"] = {
        tid: {"name": t["name"], "df": namespace[tid]}
        for tid, t in tables.items()
    }

    # -- Execute with timeout --
    exec_error: List[Exception] = []

    def _run() -> None:
        try:
            exec(compile(code, "<tabclaw_code>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            exec_error.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_EXECUTION_TIMEOUT)

    if thread.is_alive():
        return {"text": f"⛔ Execution timed out ({_EXECUTION_TIMEOUT}s limit)."}

    output_text = "".join(output_lines).rstrip()

    if exec_error:
        error_msg = f"⛔ Runtime error: {exec_error[0]}"
        return {"text": (output_text + "\n\n" + error_msg).strip()}

    # Return result DataFrame if the user assigned one
    result_val = namespace.get("result")
    if isinstance(result_val, pd.DataFrame):
        # Zero-row circuit breaker: if filtering produced an empty DataFrame,
        # stop immediately and report back instead of letting Agent hallucinate.
        if len(result_val) == 0 and _is_filter_code(code):
            return {
                "text": (
                    "⚠️ 过滤结果为 0 行：根据您的条件，未在数据集中过滤到任何符合的数据。"
                    "请检查账期、日期格式或筛选名称是否与原始数据一致。"
                ),
                "zero_row_breaker": True,
            }
        return {
            "df": result_val,
            "name": result_name,
            "text": output_text or (
                f"DataFrame assigned to 'result': "
                f"{len(result_val)} rows × {len(result_val.columns)} cols"
            ),
        }

    return {
        "text": output_text
        or "(code ran with no output — assign a DataFrame to 'result' to create a table)"
    }


def _is_filter_code(code: str) -> bool:
    """Heuristic: does the code look like a filter/select/query operation?"""
    filter_markers = [
        ".query(", ".loc[", ".iloc[", "==", "!=", ">=", "<=",
        ".filter(", ".isin(", ".str.contains(", ".between(",
        "筛选", "过滤", "filter",
    ]
    return any(marker in code for marker in filter_markers)
