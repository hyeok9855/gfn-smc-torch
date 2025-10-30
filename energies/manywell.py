from typing import Callable

import numpy as np
import torch
import torch.distributions as D

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.plot_utils import viz_2d_slice, viz_energy_hist


def rejection_sampling(
    n_samples: int, proposal: D.Distribution, target_unnormed_logp: Callable, k: float
) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample(torch.Size((n_samples * 10,)))
    u_0 = D.Uniform(0, k * torch.exp(proposal.log_prob(z_0))).sample().to(z_0)
    accept = torch.exp(target_unnormed_logp(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_unnormed_logp, k)
        samples = torch.concat([samples, new_samples], dim=0)
        return samples


class ManyWell(BaseEnergy):
    """
    log p(x1, x2) = -(x1**4) + 6 * x1**2 + 1 / 2 * x1 - (1 / 2) * (x2**2)
    """

    def __init__(
        self,
        device: str | torch.device,
        ndim=32,
        seed: int = 0,
    ) -> None:
        super().__init__(device=device, ndim=ndim, seed=seed, plot_bound=3.0)

        assert ndim % 2 == 0
        self.n_wells = ndim // 2

        # rejection sampling proposal
        component_mix = torch.tensor([0.2, 0.8]).to(self.device)
        means = torch.tensor([-1.7, 1.7]).to(self.device)
        scales = torch.tensor([0.5, 0.5]).to(self.device)
        mix = D.Categorical(component_mix)
        com = D.Normal(means, scales)
        self.proposal_x1 = D.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

        self.Z_x1 = 11784.50927
        self.Z_x2 = np.sqrt(2 * np.pi)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._manywell_unnormed_logp(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            samples = torch.cat(
                [self._sample_doublewell(batch_size) for _ in range(self.n_wells)], dim=-1
            )
        samples = samples.to(self.device)
        return samples

    def gt_logz(self) -> float:
        return self.n_wells * (np.log(self.Z_x1) + np.log(self.Z_x2))

    # ----- Energy-specific methods ----- #
    def _manywell_unnormed_logp(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2  # [batch_size, ndim]
        x_reshaped = x.view(-1, self.n_wells, 2).reshape(-1, 2)  # [batch_size * n_wells, 2]
        unnormed_logp = self._doublewell_unnormed_logp(x_reshaped)  # [batch_size * n_wells]
        unnormed_logp = unnormed_logp.reshape(-1, self.n_wells).sum(dim=1)  # [batch_size]
        return unnormed_logp

    def _sample_doublewell(self, batch_size: int) -> torch.Tensor:
        x1 = rejection_sampling(
            batch_size, self.proposal_x1, self._target_unnormed_logp_x1, self.Z_x1 * 3
        )
        x2 = torch.randn_like(x1)
        return torch.stack([x1, x2], dim=1)

    def _doublewell_unnormed_logp(self, x: torch.Tensor) -> torch.Tensor:
        return self._target_unnormed_logp_x1(x[:, 0]) + self._target_unnormed_logp_x2(x[:, 1])

    @staticmethod
    def _target_unnormed_logp_x1(x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 1
        return -(x**4) + 6 * x**2 + 1 / 2 * x

    @staticmethod
    def _target_unnormed_logp_x2(x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 1
        return -(1 / 2) * (x**2)

    def visualize(
        self, samples: torch.Tensor, weights: torch.Tensor | None = None, **kwargs
    ) -> dict:
        lim = self.plot_bound
        out_dict = {}
        for idx1, idx2 in [(0, 2), (1, 2)]:
            out_dict.update(viz_2d_slice(self, (idx1, idx2), samples, weights=weights, lim=lim))

        out_dict.update(viz_energy_hist(self, samples))
        return out_dict
