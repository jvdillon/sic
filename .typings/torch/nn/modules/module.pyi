"""Stub for torch.nn.modules.module with properly typed Module.__init__."""

from typing import Any

class Module:
    """Module with properly typed __init__ method."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
