"""
As the project grows (normalisation, smoothing, the static and dynamic classifiers),
each new step can show up in the Vision section's "Pipeline stages" tab by adding a
module to signvoice/ui/stage_plugins/:

    @register_stage("Static classifier")
    class StaticClassifierStage:
        def __init__(self): ...                       # fresh state for every camera run
        def process(self, ctx: FrameContext) -> dict: # called once per processed frame
            return {"letter": "A", "confidence": 0.93}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from signvoice.ui.registry import discover
from signvoice.ui.runtime import VisionSettings
from signvoice.vision.capture import Frame

STAGES_PACKAGE = "signvoice.ui.stage_plugins"


@dataclass
class FrameContext:
    frame: Frame
    result: Any                   
    settings: VisionSettings


_FACTORIES: dict[str, Callable[[], Any]] = {}


def register_stage(name: str):
    """Class decorator. The class is instantiated once per camera run, so it can keep state."""
    def wrap(cls):
        cls.name = name
        _FACTORIES[name] = cls
        return cls
    return wrap


def build_stages() -> list:
    discover(STAGES_PACKAGE)
    return [_FACTORIES[name]() for name in sorted(_FACTORIES)]


def run_stages(stages: list, ctx: FrameContext) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stage in stages:
        try:
            out[stage.name] = dict(stage.process(ctx))
        except Exception as exc:                       # a broken stage must not kill the camera loop
            out[stage.name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out