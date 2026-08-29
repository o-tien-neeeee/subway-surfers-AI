"""Codebase hygiene: no bare excepts, no silent passes, no stub markers.

Uses AST (not grep) so comments/strings cannot produce false positives.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))


def file_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


class TestNoBareExcepts:
    @pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
    def test_no_bare_except(self, path: Path) -> None:
        tree = file_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, (
                    f"{path.name}:{node.lineno} bare `except:` is forbidden"
                )

    @pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
    def test_no_silent_exception_pass(self, path: Path) -> None:
        tree = file_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                    assert not all(isinstance(b, ast.Pass) for b in node.body), (
                        f"{path.name}:{node.lineno} `except Exception: pass` "
                        f"swallows errors silently"
                    )

    @pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
    def test_every_except_logs_or_returns(self, path: Path) -> None:
        """Every `except Exception` must do something observable (log/raise/
        return a value) — checking it's not just a comment body."""
        tree = file_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                has_content = any(
                    isinstance(b, (ast.Raise, ast.Return, ast.Assign, ast.Expr,
                                   ast.If, ast.For, ast.While, ast.With, ast.Try,
                                   ast.Break, ast.Continue))
                    for b in node.body
                )
                assert has_content or len(node.body) > 1, (
                    f"{path.name}:{node.lineno} suspiciously empty handler"
                )


class TestNoStubs:
    # tokens assembled at runtime so this file does not match itself
    FORBIDDEN = ("TO" + "DO", "FIX" + "ME", "XX" + "X:", "NotImplement" + "ed")

    @pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
    def test_no_todo_or_ellipsis_stubs(self, path: Path) -> None:
        src = path.read_text(encoding="utf-8")
        for token in self.FORBIDDEN:
            assert token not in src, f"{path.name} contains {token!r}"
        tree = file_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body = node.body
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    pytest.fail(f"{path.name}:{node.lineno} function {node.name} is a stub")
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and node.value.value is Ellipsis:
                pytest.fail(f"{path.name}:{node.lineno} ellipsis placeholder")


class TestDependencyDiscipline:
    def test_requirements_is_minimal(self) -> None:
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        allowed = {"torch", "numpy", "opencv-python", "opencv-python-headless",
                   "pillow", "mss", "pyautogui", "pygetwindow", "pynput",
                   "psutil", "pytest", "six"}
        for line in req.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(">=")[0].split("==")[0].split("<=")[0].strip()
            assert name.lower() in allowed, f"undeclared-heavy dep: {name}"

    def test_no_network_calls_in_core(self) -> None:
        """The bot must run fully offline; core modules never fetch anything."""
        import re

        net = re.compile("reque" + "sts|urllib|http" + "x|http\\.client|"
                         "socket\\.connect|urlopen|download")
        for path in sorted(ROOT.glob("*.py")):
            src = path.read_text(encoding="utf-8")
            for i, line in enumerate(src.splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                assert not net.search(line), (
                    f"{path.name}:{i} possible network call: {line.strip()}"
                )
