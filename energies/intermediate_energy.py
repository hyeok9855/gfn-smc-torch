import torch

from energies.base import BaseEnergy
from models.gfn import GFN


class IntermediateEnergy(BaseEnergy):
    def __init__(self, target_energy: BaseEnergy, gfn: GFN, step: int) -> None:
        super().__init__(
            device=target_energy.device,
            ndim=target_energy.ndim,
            seed=target_energy.seed,
            plot_bound=target_energy.plot_bound,
        )
        self.target_energy = target_energy
        self.gfn = gfn
        self.step = step
        self.t = step / gfn.num_steps

    def energy(self, states: torch.Tensor) -> torch.Tensor:
        return -self.log_reward(states)

    def log_reward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (bsz, ndim)

        if self.t == 1.0:
            log_fs = self.target_energy.log_reward(states)
            return log_fs

        with torch.no_grad():
            _, _, log_fs = self.gfn.pred_module.forward(
                states, self.t, self.target_energy.grad_log_reward
            )
            if self.gfn.partial_energy:
                log_fs += self.gfn.get_partial_energy(
                    states.unsqueeze(1),
                    torch.tensor([self.step], device=self.device, dtype=torch.long),
                ).squeeze(1)
        return log_fs

    def visualize(self, samples: torch.Tensor, **kwargs) -> dict:
        # TODO: Implement visualization for intermediate energy and samples
        raise NotImplementedError
