"""
This module contains utility functions for the energy function of particle system.
(e.g., Lennard-Jones, Double-Well, or protein molecules.)
"""

import torch


def interatomic_distance(
    x: torch.Tensor,
    n_particles: int,
    n_dimensions: int,
    remove_duplicates: bool = True,
):
    """
    Computes the distances between all the particle pairs.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in n_dimensions.
        Tensor of shape `[n_batch, n_particles * n_dimensions]`.
    n_particles : int
        Number of particles.
    n_dimensions : int
        Number of dimensions.
    remove_duplicates : bool, optional
        Flag indicating whether to remove duplicate distances.
        If False, the all-distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration.
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles, n_particles]`.
    """

    batch_shape = x.shape[:-1]
    x = x.view(-1, n_particles, n_dimensions)

    distances = torch.cdist(x, x, p=2)

    if remove_duplicates:
        distances = distances[
            :, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1
        ]
        distances = distances.reshape(*batch_shape, n_particles * (n_particles - 1) // 2)
    else:
        distances = distances.reshape(*batch_shape, n_particles, n_particles)

    return distances


def remove_mean(x: torch.Tensor, n_particles: int, spatial_dim: int):
    """
    Removes the mean of the input tensor x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape `[n_batch, n_particles * n_dimensions]`.
    n_particles : int
        Number of particles.
    spatial_dim : int
        Spatial dimension.

    Returns
    -------
    x : torch.Tensor
        Input tensor with mean removed.
    """
    batch_shape = x.shape[:-1]
    x = x.reshape(*batch_shape, n_particles, spatial_dim)
    x = x - x.mean(dim=-2, keepdim=True)
    return x.reshape(*batch_shape, n_particles * spatial_dim)
