"""
registry.py - the plug-in point for top-level menu sections.

A section is one entry in the centre menu (Vision, LLM, TTS, Project, ...). Each
lives in its own module inside signvoice/ui/sections/ and registers itself:

    register_section(Section(key="llm", label="LLM", icon="X", blurb="...", render=render))

`render=None` means "not built yet": the menu shows the button greyed out. When the
feature is ready, write a `render()` function and pass it in. Nothing else changes.

Every module in the sections package is imported automatically (see discover), so
adding a section means adding a file. A module that fails to import is reported in
the app instead of taking the whole page down.
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

SECTIONS_PACKAGE = "signvoice.ui.sections"


@dataclass(frozen=True)
class Section:
    key: str                                   # unique id, used for session state and widget keys
    label: str                                 # text on the menu button
    icon: str                                  # short emoji shown before the label
    blurb: str                                 # one-line description (tooltip and page caption)
    render: Optional[Callable[[], None]] = None
    order: int = 100                           # menu position, lowest first

    @property
    def enabled(self) -> bool:
        return self.render is not None


_SECTIONS: dict[str, Section] = {}
_LOAD_ERRORS: dict[str, str] = {}


def register_section(section: Section) -> Section:
    """Add (or replace) a section. Registering the same key twice just overwrites it."""
    _SECTIONS[section.key] = section
    return section


def discover(package: str) -> None:
    """Import every module inside `package` so that it can register itself. Safe to repeat."""
    try:
        pkg = importlib.import_module(package)
    except Exception:
        _LOAD_ERRORS[package] = traceback.format_exc()
        return
    for info in pkgutil.iter_modules(pkg.__path__):
        name = f"{package}.{info.name}"
        try:
            importlib.import_module(name)
            _LOAD_ERRORS.pop(name, None)
        except Exception:
            _LOAD_ERRORS[name] = traceback.format_exc()


def get_sections() -> list[Section]:
    discover(SECTIONS_PACKAGE)
    return sorted(_SECTIONS.values(), key=lambda s: (s.order, s.label))


def load_errors() -> dict[str, str]:
    """Modules that failed to import, mapped to their traceback."""
    return dict(_LOAD_ERRORS)