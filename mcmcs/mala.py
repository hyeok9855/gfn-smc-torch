from typing import TYPE_CHECKING

import torch
import numpy as np
from tqdm import trange
from mcmcs.base import BaseMCMC

if TYPE_CHECKING:
    from energies import BaseEnergy


class MALA(BaseMCMC):
    def __init__(
        self,
        energy: "BaseEnergy",
        burn_in: int = 100,
        n_steps: int = 1000,
        step_size: float = 0.1,
        ld_schedule: bool = True,
        target_acceptance_rate: float = 0.574,
        **kwargs,
    ):
        super().__init__(energy)
        self.burn_in = burn_in
        self.n_steps = n_steps
        self.step_size = step_size
        self.ld_schedule = ld_schedule
        self.target_acceptance_rate = target_acceptance_rate

    def adjust_ld_step(
        self,
        ld_step: float,
        acceptance_rate: float,
        adjustment_factor: float = 0.01,
    ):
        if acceptance_rate > self.target_acceptance_rate:
            return ld_step + adjustment_factor * ld_step
        else:
            return ld_step - adjustment_factor * ld_step

    def sample(self, x):
        accepted_samples = []
        accepted_logr = []
        acceptance_rate_lst = []
        log_r_original = self.energy.log_reward(x)
        acceptance_count = 0
        acceptance_rate = 0
        total_proposals = 0

        for i in trange(self.n_steps, desc="[MALA]", dynamic_ncols=True):
            x = x.requires_grad_(True)

            log_rs = self.energy.log_reward(x)
            r_grad_original = torch.autograd.grad(log_rs.sum(), x)[0].detach()
            if self.ld_schedule:
                ld_step = (
                    self.step_size if i == 0 else self.adjust_ld_step(ld_step, acceptance_rate)
                )
            else:
                ld_step = self.step_size

            new_x = x + ld_step * r_grad_original + np.sqrt(2 * ld_step) * torch.randn_like(x)
            log_r_new = self.energy.log_reward(new_x)
            r_grad_new = torch.autograd.grad(log_r_new.sum(), new_x)[0].detach()

            with torch.no_grad():
                log_q_fwd = -(
                    torch.norm(new_x - x - ld_step * r_grad_original, p=2, dim=1) ** 2
                ) / (4 * ld_step)
                log_q_bwd = -(torch.norm(x - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (
                    4 * ld_step
                )

                log_accept = (log_r_new - log_r_original) + log_q_bwd - log_q_fwd
                accept_mask = torch.rand(x.shape[0], device=x.device) < torch.exp(
                    torch.clamp(log_accept, max=0)
                )
                acceptance_count += accept_mask.sum().item()
                total_proposals += x.shape[0]

                x = x.detach()
                # After burn-in process
                if i > self.burn_in:
                    accepted_samples.append(new_x[accept_mask].detach())
                    accepted_logr.append(log_r_new[accept_mask].detach())
                x[accept_mask] = new_x[accept_mask]
                log_r_original[accept_mask] = log_r_new[accept_mask]

                if i % 5 == 0:
                    acceptance_rate = acceptance_count / total_proposals
                    if i > self.burn_in:
                        acceptance_rate_lst.append(acceptance_rate)
                    acceptance_count = 0
                    total_proposals = 0

        xs = torch.cat(accepted_samples, dim=0)
        log_rs = torch.cat(accepted_logr, dim=0)
        return xs, log_rs
