import warnings
from functools import partial
from typing import Callable, Literal

import torch


def multinomial(weights: torch.Tensor, N: int, replacement: bool = True) -> torch.Tensor:
    """Return sampled indices from multinomial distribution.

    Args:
        weights: torch.Tensor of shape (bs,)
        N: int
        replacement: bool
    """
    return torch.multinomial(weights, N, replacement=replacement)


def stratified(weights: torch.Tensor, N: int, replacement: bool = True) -> torch.Tensor:
    """Return sampled indices using stratified (re)sampling technique.

    Args:
        weights: torch.Tensor of shape (bs,)
        N: int
    """
    if not replacement:
        warnings.warn(
            "Stratified sampling does not support sampling without replacement. "
            "Using multinomial sampling instead."
        )
        return multinomial(weights, N, replacement=True)

    # Normalize weights
    weights = weights / weights.sum()

    cumsum = torch.cumsum(weights, dim=0)
    u = torch.arange(N, device=weights.device, dtype=torch.float32)
    u = (u + torch.rand(N, device=weights.device)) / N
    u = torch.searchsorted(cumsum, u).clamp(0, len(weights) - 1)
    return u


def systematic(weights: torch.Tensor, N: int, replacement: bool = True) -> torch.Tensor:
    """Return sampled indices using systematic (re)sampling technique.

    Args:
        weights: torch.Tensor of shape (bs,)
        N: int
    """
    if not replacement:
        warnings.warn(
            "Systematic sampling does not support sampling without replacement. "
            "Using multinomial sampling instead."
        )
        return multinomial(weights, N, replacement=True)

    # Normalize weights if they're not already normalized
    weights_sum = weights.sum()
    if not torch.isclose(weights_sum, torch.tensor(1.0, device=weights.device)):
        weights = weights / weights_sum

    cumsum = torch.cumsum(weights, dim=0)
    u = torch.arange(N, device=weights.device, dtype=torch.float32)
    u = (u + torch.rand(1, device=weights.device)) / N
    u = torch.searchsorted(cumsum, u).clamp(0, len(weights) - 1)
    return u


def rank(
    weights: torch.Tensor, N: int, replacement: bool = True, rank_k: float = 0.01
) -> torch.Tensor:
    """Return sampled indices using rank-based (re)sampling technique.

    Args:
        weights: torch.Tensor of shape (bs,)
        N: int
        replacement: bool
        rank_k: float
    """
    ranks = torch.argsort(torch.argsort(-weights))
    weights = 1.0 / (rank_k * len(weights) + ranks)
    return multinomial(weights, N, replacement=replacement)


def get_sampling_func(
    sampling_strategy: Literal["multinomial", "stratified", "systematic", "rank"],
    rank_k: float = 0.01,
) -> Callable[[torch.Tensor, int, bool], torch.Tensor]:
    if sampling_strategy == "multinomial":
        return multinomial
    elif sampling_strategy == "stratified":
        return stratified
    elif sampling_strategy == "systematic":
        return systematic
    elif sampling_strategy == "rank":
        return partial(rank, rank_k=rank_k)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
