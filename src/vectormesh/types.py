from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn


class VectorMeshError(Exception):
    """
    Base exception for all VectorMesh errors.

    Includes educational hints and fixes to help users understand
    and resolve tensor flow and composition issues.

    Args:
        message: Primary error message
        hint: Educational hint about what went wrong
        fix: Suggested fix or next steps
    """

    def __init__(
        self, message: str, hint: Optional[str] = None, fix: Optional[str] = None
    ):
        super().__init__(message)
        self.hint = hint
        self.fix = fix


class Cachable(BaseModel):
    """
    Base class for cachable components.
    Enforces strict validation and immutable configuration using Pydantic.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


TensorInput = Union[Float[Tensor, "..."], Tuple[Float[Tensor, "..."], ...]]
"""Documented union for connector/wiring code that accepts either a single tensor
or a tuple of tensors. Concrete components narrow to one branch and enforce it
at runtime via @jaxtyped(typechecker=beartype)."""


class BaseComponent(nn.Module, ABC):
    """Root class for all pipeline components."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Float[Tensor, "..."]: ...
