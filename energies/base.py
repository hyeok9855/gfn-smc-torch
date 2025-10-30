import abc

import torch


class BaseEnergy(abc.ABC):
    is_particle_system = False

    def __init__(
        self,
        device: str | torch.device,
        ndim: int,
        seed: int = 0,
        plot_bound: float = 1.0,
    ) -> None:
        self.device = device
        self.ndim = ndim
        self.seed = seed
        self.plot_bound = plot_bound
        self.gt_xs: torch.Tensor | None = None
        self.gt_xs_log_rewards: torch.Tensor | None = None

    @abc.abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        log_r = -self.energy(x)
        return log_r

    def grad_log_reward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.log_reward(copy_x).sum().backward()
                lgv = copy_x.grad
                assert lgv is not None
        return lgv.data

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        raise NotImplementedError

    def gt_logz(self) -> float:
        raise NotImplementedError

    def cached_sample(
        self, batch_size: int, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gt_xs is None or batch_size != self.gt_xs.size(0):
            self.gt_xs = self.sample(batch_size, seed)
            self.gt_xs_log_rewards = self.log_reward(self.gt_xs)
        assert self.gt_xs_log_rewards is not None
        return self.gt_xs, self.gt_xs_log_rewards

    def visualize(self, samples: torch.Tensor, **kwargs) -> dict:
        raise NotImplementedError
