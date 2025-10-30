import numpy as np
import torch
import torch.distributions as D

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.plot_utils import viz_2d_slice, viz_energy_hist


class Funnel(BaseEnergy):
    """
    x0 ~ N(0, sigma^2), xi | x0 ~ N(0, exp(x0)), i = 1, ..., 9
    """

    def __init__(
        self,
        device: str | torch.device,
        ndim=10,
        sigma: float = 3.0,
        seed: int = 0,
    ) -> None:
        super().__init__(device=device, ndim=ndim, seed=seed, plot_bound=10.0)
        self.dist_dominant = D.Normal(
            torch.tensor([0.0], device=self.device), torch.tensor([sigma], device=self.device)
        )
        self.mean_other = torch.zeros(self.ndim - 1, device=self.device).float()
        self.cov_eye = (
            torch.eye(self.ndim - 1, device=self.device)
            .float()
            .view(1, self.ndim - 1, self.ndim - 1)
        )

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._funnel_logprob(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            dominant_x = self.dist_dominant.sample((batch_size,)).to(self.device)  # (B, 1)
            x_others = self._dist_other(dominant_x).sample()  # (B, dim-1)
        samples = torch.hstack([dominant_x, x_others])
        samples = samples.to(self.device)
        return samples

    def gt_logz(self) -> float:
        return 0.0

    # ----- Energy-specific methods ----- #
    def _funnel_logprob(self, x: torch.Tensor) -> torch.Tensor:
        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )

        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = torch.exp(x[:, 0:1])
        neg_log_density_other = 0.5 * np.log(2 * np.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        log_density_other = torch.sum(-neg_log_density_other, dim=-1)

        return log_density_dominant + log_density_other

    def _dist_other(self, dominant_x: torch.Tensor) -> D.MultivariateNormal:
        variance_other = torch.exp(dominant_x)
        cov_other = variance_other.view(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return D.multivariate_normal.MultivariateNormal(self.mean_other, cov_other)

    def visualize(self, samples, weights: torch.Tensor | None = None, **kwargs):
        lim = self.plot_bound
        out_dict = {}
        for i in range(1, 3):
            out_dict.update(viz_2d_slice(self, (0, i), samples, weights=weights, lim=lim))
        out_dict.update(viz_energy_hist(self, samples))
        return out_dict
