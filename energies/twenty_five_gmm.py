import torch
import torch.distributions as D

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.plot_utils import viz_2d_slice, viz_energy_hist


class TwentyFiveGaussianMixture(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device,
        seed: int = 0,
    ) -> None:
        ndim = 2
        super().__init__(device=device, ndim=ndim, seed=seed, plot_bound=13.0)

        self.nmode = 25
        modes = torch.Tensor([(a, b) for a in [-10, -5, 0, 5, 10] for b in [-10, -5, 0, 5, 10]]).to(
            self.device
        )

        self.gmm = [
            D.MultivariateNormal(
                loc=mode, covariance_matrix=0.3 * torch.eye(ndim, device=self.device)
            )
            for mode in modes
        ]

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._log_prob(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            samples = torch.cat(
                [mvn.sample(torch.Size((batch_size // self.nmode,))) for mvn in self.gmm], dim=0
            )
        samples = samples.to(self.device)
        return samples

    def gt_logz(self) -> float:
        return 0.0

    # ----- Energy-specific methods ----- #
    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(
            torch.stack([mvn.log_prob(x) for mvn in self.gmm]), dim=0, keepdim=False
        ) - torch.log(torch.tensor(self.nmode, device=self.device))

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
