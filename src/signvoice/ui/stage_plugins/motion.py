"""
Example stage: how much each hand moved since the previous processed frame.

Also a first look at a signal the project will need later: a letter is a held handshape
(little motion) while a word sign moves. Values depend on the processing frame rate, because
slow processing skips frames and makes the gap between compared frames larger.
"""
from __future__ import annotations

import numpy as np

from signvoice.ui.stages import FrameContext, register_stage
from signvoice.vision.landmark import LEFT_HAND, RIGHT_HAND


@register_stage("Hand motion")
class MotionStage:
    def __init__(self) -> None:
        self._prev = None

    def process(self, ctx: FrameContext) -> dict:
        f = ctx.result.features
        current = {"left": (f[LEFT_HAND], ctx.result.left_found),
                   "right": (f[RIGHT_HAND], ctx.result.right_found)}
        out = {}
        for side, (coords, found) in current.items():
            prev = self._prev.get(side) if self._prev else None
            if found and prev is not None:
                out[f"{side} hand (mean change)"] = round(float(np.abs(coords - prev).mean()), 4)
            else:
                out[f"{side} hand (mean change)"] = "n/a"
        self._prev = {side: coords.copy() if found else None for side, (coords, found) in current.items()}
        return out