"""Headless guard against misspelled Tkinter attribute names.

Why this file exists
--------------------
``gui.py`` shipped ``self.root.winfovrootheight()`` -- the underscores were
missing -- for a whole release.  It was *not* Windows-specific: accepting a
screen region raises ``AttributeError`` from inside a Tkinter callback on
every platform.  It survived a 449-test suite for one reason only: not a
single test ever constructs a Tk root, because CI has no display, so the
whole GUI layer was executed zero times.

A syntax check cannot help here -- ``winfovrootheight`` is a perfectly valid
identifier, it just does not exist.  Neither can pyflakes: it is an
*attribute* of a live object, not an undefined name.

So the check has to compare the attribute names ``gui.py`` references against
Tkinter's real API surface.  Two sources, in order of preference:

1.  The installed ``tkinter`` module itself.  Authoritative, and this is what
    runs on a normal developer machine.
2.  A vendored list transcribed from CPython 3.13 ``Lib/tkinter``.  Used when
    ``tkinter`` cannot be imported at all (a slim Linux container without
    ``python3-tk``), which is exactly the environment where the bug hid.

Crucially this module only *parses* ``gui.py`` with ``ast``.  It never imports
it, so it needs no display, no ``python3-tk`` and no ``DISPLAY`` variable, and
it therefore runs -- and fails loudly -- everywhere.
"""
from __future__ import annotations

import ast
import pathlib
import sys
from typing import Iterable

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUI = ROOT / "gui.py"

# --------------------------------------------------------------------------- #
# Source 1: the real Tkinter, when it is importable.
# --------------------------------------------------------------------------- #
def _live_api() -> set[str] | None:
    """Every attribute a Tk widget can expose, from the installed module."""
    try:
        import tkinter
        from tkinter import ttk
    except Exception:                                  # no Tk in this interpreter
        return None
    classes: list[type] = [
        tkinter.Misc, tkinter.Tk, tkinter.Toplevel, tkinter.Widget,
        tkinter.BaseWidget, tkinter.Pack, tkinter.Grid, tkinter.Place,
        tkinter.Canvas, tkinter.Frame, tkinter.Label, tkinter.Entry,
        tkinter.Button, tkinter.Checkbutton, tkinter.Scrollbar,
        tkinter.Text, tkinter.Listbox, tkinter.OptionMenu, tkinter.Spinbox,
        tkinter.LabelFrame, tkinter.PanedWindow, tkinter.Menu,
        tkinter.StringVar, tkinter.IntVar, tkinter.DoubleVar, tkinter.BooleanVar,
        ttk.Frame, ttk.Label, ttk.Button, ttk.Entry, ttk.Checkbutton,
        ttk.Combobox, ttk.Style, ttk.Progressbar, ttk.Notebook, ttk.Widget,
    ]
    api: set[str] = set()
    for cls in classes:
        api |= set(dir(cls))
    return api


# --------------------------------------------------------------------------- #
# Source 2: vendored fallback, transcribed from CPython 3.13 Lib/tkinter.
#
# Two parts.  The first is every widget method gui.py currently reaches, so
# any *new* name has to be reviewed before it lands.  The second is the whole
# winfo_*/wm_* family, so a fresh typo in either family is caught even though
# gui.py does not use all of them today.
# --------------------------------------------------------------------------- #
_VENDORED_WIDGET_METHODS = {
    "after", "attributes", "bind", "create_rectangle", "create_text",
    "delete", "destroy", "focus_force", "geometry", "itemconfigure",
    "mainloop", "pack", "protocol", "set", "state", "title",
    "update_idletasks",
}

_VENDORED_WINFO_WM_FAMILY = {
    "winfo_atom", "winfo_atomname", "winfo_cells", "winfo_children",
    "winfo_class", "winfo_colormapfull", "winfo_containing", "winfo_depth",
    "winfo_exists", "winfo_fpixels", "winfo_geometry", "winfo_height",
    "winfo_id", "winfo_interps", "winfo_ismapped", "winfo_manager",
    "winfo_name", "winfo_parent", "winfo_pathname", "winfo_pixels",
    "winfo_pointerx", "winfo_pointerxy", "winfo_pointery", "winfo_reqheight",
    "winfo_reqwidth", "winfo_rgb", "winfo_rootx", "winfo_rooty",
    "winfo_screen", "winfo_screencells", "winfo_screendepth",
    "winfo_screenheight", "winfo_screenmmheight", "winfo_screenmmwidth",
    "winfo_screenvisual", "winfo_screenwidth", "winfo_server",
    "winfo_toplevel", "winfo_viewable", "winfo_visual", "winfo_visualid",
    "winfo_visualsavailable", "winfo_vrootheight", "winfo_vrootwidth",
    "winfo_vrootx", "winfo_vrooty", "winfo_width", "winfo_x", "winfo_y",
    "wm_aspect", "wm_attributes", "wm_client", "wm_colormapwindows",
    "wm_command", "wm_deiconify", "wm_focusmodel", "wm_forget", "wm_frame",
    "wm_geometry", "wm_grid", "wm_group", "wm_iconbitmap", "wm_iconify",
    "wm_iconmask", "wm_iconname", "wm_iconphoto", "wm_iconposition",
    "wm_iconwindow", "wm_manage", "wm_maxsize", "wm_minsize",
    "wm_overrideredirect", "wm_positionfrom", "wm_protocol", "wm_resizable",
    "wm_sizefrom", "wm_state", "wm_title", "wm_transient", "wm_withdraw",
}

_VENDORED = _VENDORED_WIDGET_METHODS | _VENDORED_WINFO_WM_FAMILY


def _api() -> tuple[set[str], str]:
    live = _live_api()
    if live is not None:
        return live, "installed tkinter"
    return _VENDORED, "vendored CPython 3.13 list"


# --------------------------------------------------------------------------- #
# Which attribute accesses in gui.py are Tk calls?
#
# gui.py holds its widgets in a small, stable set of names.  Anything reached
# through one of those, plus anything in the winfo_*/wm_* namespaces, is
# treated as a Tk call.  Being generous here is safe: an unknown name is a
# real bug either way.
# --------------------------------------------------------------------------- #
_TK_OWNERS = {
    "root", "top", "win", "overlay", "canvas", "label", "hint", "status",
    "lbl", "sel", "region_var",
    "self.root", "self.top", "self.win", "self.overlay", "self.canvas",
    "self.label", "self.hint", "self.status", "self.lbl", "self.sel",
    "self.region_var",
}


def _owner_of(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _tk_attribute_uses(source: str) -> list[tuple[str, int]]:
    """Every (attribute, lineno) in *source* that looks like a Tk call."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
            continue
        if _owner_of(node.value) in _TK_OWNERS or node.attr.startswith(("winfo", "wm_")):
            found.append((node.attr, node.lineno))
    return found


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestTkApiSurface:
    """Every Tkinter attribute gui.py touches must actually exist."""

    def test_gui_references_only_real_tkinter_attributes(self) -> None:
        """The regression guard for the ``winfovrootheight`` crash.

        Read the message: on failure it prints the offending name *and* the
        nearest real spelling, so the fix is obvious rather than another
        debugging session.
        """
        api, source = _api()
        unknown = sorted(
            {name for name, _ in _tk_attribute_uses(GUI.read_text(encoding="utf-8"))
             if name not in api}
        )
        assert not unknown, (
            f"gui.py calls attribute(s) that do not exist in Tkinter "
            f"(checked against {source}): {unknown}.  "
            f"Near matches: "
            + "; ".join(f"{u!r} -> {self._nearest(u, api)!r}" for u in unknown)
            + ".  These fail only at runtime, from inside a Tk callback, on "
              "every platform -- see the module docstring."
        )

    @staticmethod
    def _nearest(name: str, api: Iterable[str]) -> str:
        import difflib
        matches = difflib.get_close_matches(name, sorted(api), n=1, cutoff=0.5)
        return matches[0] if matches else "(nothing similar)"

    def test_the_guard_actually_catches_the_historical_typo(self) -> None:
        """Prove the guard is not vacuous.

        A linter that passes on the broken input is worse than no linter: it
        manufactures confidence.  Re-introduce the exact historical typo and
        require the scan to flag it, by name and line.
        """
        api, _ = _api()
        broken = GUI.read_text(encoding="utf-8").replace(
            "winfo_vrootheight()", "winfovrootheight()"
        )
        assert broken != GUI.read_text(encoding="utf-8"), (
            "the fixture no longer contains winfo_vrootheight(); this test "
            "cannot prove anything until gui.py uses that call again"
        )
        flagged = {n for n, _ in _tk_attribute_uses(broken) if n not in api}
        assert "winfovrootheight" in flagged

    def test_the_correct_spelling_is_accepted(self) -> None:
        """Guard against the inverse failure: the guard rejecting valid code."""
        api, _ = _api()
        for name in ("winfo_vrootheight", "winfo_vrootwidth", "winfo_fpixels",
                     "geometry", "pack", "protocol", "title", "attributes"):
            assert name in api, f"{name} is a real Tkinter attribute"

    def test_gui_still_parses(self) -> None:
        """Cheap precondition: the scan above is only as good as the parse."""
        ast.parse(GUI.read_text(encoding="utf-8"))

    def test_both_width_and_height_are_read(self) -> None:
        """The original bug was asymmetric: width was spelled right, height
        was not.  Pin the pair so a future edit cannot reintroduce that
        one-sided failure mode.
        """
        uses = {n for n, _ in _tk_attribute_uses(GUI.read_text(encoding="utf-8"))}
        assert "winfo_vrootwidth" in uses and "winfo_vrootheight" in uses

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="ast.parse needs 3.8+")
    def test_no_attribute_name_is_a_run_on_of_another(self) -> None:
        """Catch the general shape of the bug, not just this instance.

        ``winfovrootheight`` is ``winfo_vrootheight`` with the underscores
        deleted.  Any attribute whose de-underscored form collides with a
        *real* attribute is almost certainly a mangled spelling.  This is
        deliberately narrow -- it only fires on exact collisions -- so it
        cannot produce false positives.
        """
        api, _ = _api()
        by_squash = {n.replace("_", ""): n for n in api}
        offenders = [
            (name, line, by_squash[name.replace("_", "")])
            for name, line in _tk_attribute_uses(GUI.read_text(encoding="utf-8"))
            if "_" not in name
            and name.replace("_", "") in by_squash
            and name not in api
        ]
        assert not offenders, f"likely mangled spellings: {offenders}"


class TestDpiAwarenessOrdering:
    """``SetProcessDpiAwareness`` must run before the first window exists.

    Windows refuses the call once a window has been created, and the refusal
    was being swallowed.  The symptom is nasty: Tk reports virtualised screen
    coordinates while mss captures physical pixels, so the region the user
    drags out is not the region that gets captured -- on scaled desktops
    only, i.e. exactly the machines that need this.  ``gui.py`` imports
    tkinter at module level so it cannot be imported without a display; the
    ordering is therefore asserted structurally.
    """

    def _call_order_in(self, func_name: str) -> list[str]:
        tree = ast.parse(GUI.read_text(encoding="utf-8"))
        target = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == func_name),
            None,
        )
        assert target is not None, f"no function {func_name!r} in gui.py"
        names = []
        for node in ast.walk(target):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    names.append(f.id)
                elif isinstance(f, ast.Attribute):
                    names.append(f"{_owner_of(f.value)}.{f.attr}")
        return names

    def test_run_gui_sets_dpi_awareness_before_creating_tk(self) -> None:
        order = self._call_order_in("run_gui")
        assert "set_dpi_awareness" in order, "run_gui never sets DPI awareness"
        assert any(n.endswith(".Tk") for n in order), "run_gui never creates tk.Tk()"
        assert order.index("set_dpi_awareness") < next(
            i for i, n in enumerate(order) if n.endswith(".Tk")
        ), "set_dpi_awareness() must be called before tk.Tk()"

    def test_set_dpi_awareness_does_not_hardcode_a_scale_factor(self) -> None:
        """The old version returned a literal 1.0 on every path while its
        docstring claimed it returned "the Tk scale factor".  A caller that
        trusted it would treat a 150% desktop as 100%.
        """
        src = GUI.read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "set_dpi_awareness"
        )
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        assert returns, "set_dpi_awareness returns nothing at all"
        # Note: `True == 1.0` is True in Python, so a naive equality test
        # flags every `return True`.  Match on the type as well.
        literals = [
            n for n in returns
            if isinstance(n.value, ast.Constant)
            and not isinstance(n.value.value, bool)
            and n.value.value == 1.0
        ]
        assert not literals, (
            "set_dpi_awareness still returns a hardcoded 1.0, which is not a "
            "measured scale factor and must not be trusted by callers"
        )
        # It must now report success/failure, which is what the callers need.
        assert all(
            isinstance(n.value, ast.Constant) and isinstance(n.value.value, bool)
            for n in returns
        ), "every path should return a bool: was DPI awareness established?"

    def test_region_accept_no_longer_discards_the_result(self) -> None:
        """The dead `scale = set_dpi_awareness()` assignment is what let the
        too-late call go unnoticed for so long."""
        # Assert on the AST, not on the file text: the explanatory comment
        # for this very fix mentions the old expression, and a substring
        # search would reject the fix because of its own documentation.
        tree = ast.parse(GUI.read_text(encoding="utf-8"))
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_region_accepted"
        )
        bound = [
            t.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "set_dpi_awareness"
            for t in n.targets if isinstance(t, ast.Name)
        ]
        assert "scale" not in bound, (
            "the result is being thrown away into a dead `scale` variable "
            "again -- that dead assignment is what hid the bug originally"
        )
        assert bound, "_region_accepted no longer calls set_dpi_awareness()"
        # And the value must actually be *used*, not merely reassigned.
        uses = [
            n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        ]
        assert bound[0] in uses, (
            f"{bound[0]} is assigned from set_dpi_awareness() but never read, "
            "so a refused DPI request stays invisible"
        )

    def test_region_accept_reads_both_screen_dimensions(self) -> None:
        """Pin the width/height pair next to the DPI call, since that is where
        the historical typo lived."""
        src = GUI.read_text(encoding="utf-8")
        assert "winfo_vrootwidth()" in src and "winfo_vrootheight()" in src
