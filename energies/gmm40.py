import numpy as np
import torch
import torch.distributions as D

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.plot_utils import viz_2d_slice, viz_energy_hist


class GMM40(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device,
        ndim: int = 2,
        num_components: int = 40,
        loc_scaling: float = 40.0,
        scale_scaling: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(device=device, ndim=ndim, seed=seed, plot_bound=loc_scaling * 1.4)

        try:
            locs = torch.from_numpy(np.load(f"energies/data/gmm40-{ndim}d_locs.npy"))
        except FileNotFoundError:
            with temp_seed(seed):
                locs = torch.rand(num_components, ndim) * 2 - 1
        locs = locs.to(dtype=torch.float32, device=device) * loc_scaling
        scales = torch.ones(num_components, ndim, device=device) * scale_scaling
        logits = torch.ones(num_components, device=device)

        mixture_dist = D.Categorical(logits=logits)
        components_dist = D.Independent(
            D.Normal(loc=locs, scale=scales), reinterpreted_batch_ndims=1
        )
        self.distribution = D.MixtureSameFamily(mixture_dist, components_dist)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._log_prob(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            return self.distribution.sample(sample_shape=torch.Size((batch_size,))).to(self.device)

    def gt_logz(self):
        return 0.0

    # ----- Energy-specific methods ----- #
    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)

        log_prob = self.distribution.log_prob(x)

        if not batched:
            log_prob = log_prob.squeeze(0)

        return log_prob

    def visualize(
        self, samples: torch.Tensor, weights: torch.Tensor | None = None, **kwargs
    ) -> dict:
        lim = self.plot_bound
        out_dict = {}
        for i in range(1, min(self.ndim, 4), 2):
            out_dict.update(
                viz_2d_slice(
                    self, (i - 1, i), samples, weights=weights, lim=lim, n_contour_levels=100
                )
            )

        out_dict.update(viz_energy_hist(self, samples))
        return out_dict
