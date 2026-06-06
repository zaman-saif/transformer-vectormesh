from abc import abstractmethod

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class BaseAggregator(BaseComponent):
    """Base class for aggregating 3D -> 2D tensors.
    We use "forward" to be compatible with nn.Module
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Aggregate from (batch, chunks, dim) to (batch, dim)."""
        ...


class MeanAggregator(BaseAggregator):
    """Aggregate by taking mean over chunks.
    no learnable parameters.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Mean over chunks dimension."""
        return tensors.mean(dim=1)


class MaskedMeanAggregator(BaseAggregator):
    """Mean aggregator that ignores zero-padded chunks (from FixedPadding).
    Detects padding by checking if all values in a chunk are zero.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        mask = tensors.abs().sum(dim=-1) > 0  # (batch, chunks), False for padded
        mask_f = mask.unsqueeze(-1).float()  # (batch, chunks, 1)
        summed = (tensors * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1)
        return summed / count


class AttentionAggregator(BaseAggregator):
    """Aggregate using learnable attention over chunks.
    Because attention does not handle variable-length sequences,
    we actually get (batch, chunks, dim) where chunks is fixed.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        # attention_weights: (batch, _, 1)
        attention_weights = torch.softmax(self.attention(tensors), dim=1)
        return (tensors * attention_weights).sum(dim=1)


class RNNAggregator(BaseAggregator):
    """Aggregate using RNN over chunks.
    return final hidden state.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        output, _ = self.rnn(tensors)
        return output[:, -1, :]
