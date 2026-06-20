import ast
import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORBIDDEN_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}

FORBIDDEN_ATTRS = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "read",
    "readline",
    "readlines",
    "write",
    "writelines",
    "read_text",
    "write_text",
    "to_csv",
    "to_excel",
    "to_pickle",
    "to_sql",
    "read_csv",
    "read_excel",
    "read_sql",
}

FORBIDDEN_PATH_METHODS = {
    "rename",
    "replace",
    "unlink",
    "rmdir",
    "mkdir",
    "touch",
    "write_text",
    "read_text",
}


class SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError("Import is forbidden in generated code.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError("Import is forbidden in generated code.")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden function call: {node.func.id}")

        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr

            if attr_name.startswith("__") or attr_name in FORBIDDEN_ATTRS:
                raise ValueError(f"Forbidden method call: {attr_name}")

            if attr_name in FORBIDDEN_PATH_METHODS and self._looks_like_path_object(node.func.value):
                raise ValueError(f"Forbidden path method call: {attr_name}")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError("Dunder attributes are forbidden.")

        self.generic_visit(node)

    @staticmethod
    def _looks_like_path_object(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in {"output_dir", "chart_path", "path", "file_path"}

        if isinstance(node, ast.BinOp):
            return True

        return False


def sanitize_code(code: str) -> str:
    text = code.strip()

    text = re.sub(r"^```python\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    clean_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            clean_lines.append(line)
            continue

        if stripped.startswith("```"):
            continue

        if stripped.startswith("import ") or stripped.startswith("from "):
            continue

        clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()

    if not cleaned.startswith("result"):
        cleaned = "result = {}\n" + cleaned

    return cleaned


def validate_code(code: str) -> str:
    sanitized_code = sanitize_code(code)
    tree = ast.parse(sanitized_code)
    SafetyVisitor().visit(tree)
    return sanitized_code


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): make_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_jsonable(item) for item in obj]

    if isinstance(obj, tuple):
        return [make_jsonable(item) for item in obj]

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        return obj.head(50).to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.head(50).to_dict()

    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        return str(obj)


def run_generated_code(df: pd.DataFrame, code: str, output_dir: Path) -> dict[str, Any]:
    code = validate_code(code)

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "float": float,
        "int": int,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "print": print,
    }

    globals_dict = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
    }

    locals_dict = {
        "df": df.copy(),
        "output_dir": output_dir,
        "result": {},
        "charts": [],
    }

    stdout_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer):
        exec(code, globals_dict, locals_dict)

    return {
        "result": make_jsonable(locals_dict.get("result", {})),
        "charts": make_jsonable(locals_dict.get("charts", [])),
        "stdout": stdout_buffer.getvalue(),
    }