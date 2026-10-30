import inspect
from typing import Callable

def stretch(fn: Callable) -> dict:
    params = inspect.signature(fn).parameters
    if "width" in params and "stretch" in (fn.__doc__ or ""):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}