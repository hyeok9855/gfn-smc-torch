import math

import torch
from tqdm import trange

from energies import ALDP
from mcmcs.base import BaseMCMC


class MD(BaseMCMC):
    def __init__(
        self,
        energy: ALDP,
        gamma: float = 1.0,
        n_steps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        step_size: float = 0.001,
        integrator: str = "euler",
        **kwargs,
    ) -> None:
        super().__init__(energy)
        self.gamma = gamma
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.thinning = thinning

        self.dt = step_size
        self.dt_half = self.dt / 2
        self.mass = self.energy.mass
        self.kBT = self.energy.kBT
        self.beta = self.energy.beta

        self.std = torch.sqrt(2 * self.kBT * self.gamma * self.dt / self.mass)
        self.mol_ndim = self.energy.coordinate_transform.transform.n_dim
        assert self.mol_ndim % 3 == 0
        self.n_atoms = self.mol_ndim // 3

        self.integrator = integrator

        # For BAOAB
        self.exp_mgdt = math.exp(-self.gamma * self.dt)
        self.std_baoab = torch.sqrt(self.kBT * (1 - (self.exp_mgdt**2)) / self.mass)

    def sample(self, xs: torch.Tensor):
        positions = []
        log_rs = []

        bsz = xs.shape[0]
        x_position, _ = self.energy.transform(xs)

        position = x_position.reshape(bsz, self.n_atoms, 3)
        velocity = torch.zeros_like(position, device=position.device)
        position = position.requires_grad_(True)
        energy = self.energy.p.norm_energy(position.reshape(bsz, self.mol_ndim)) / self.beta
        force = -torch.autograd.grad(energy.sum(), position)[0]

        for _ in trange(self.n_steps, desc="[MD]", dynamic_ncols=True):
            if self.integrator == "euler":
                position, velocity, force, log_r = self.step_euler(position, velocity, force)
            else:  # baoab
                position, velocity, force, log_r = self.step_baoab(position, velocity, force)
            positions.append(position)
            log_rs.append(log_r)

        # stack after burning in first self.burn_in positions and rewards
        positions = torch.stack(positions[self.burn_in :], dim=0)
        log_rs = torch.stack(log_rs[self.burn_in :], dim=0)

        positions = positions.reshape(-1, self.mol_ndim)
        log_rs = log_rs.reshape(-1)

        new_xs, log_det = self.energy.inverse(positions)
        log_rs = log_rs - log_det

        new_xs = new_xs.reshape(-1, bsz, self.energy.ndim)
        log_rs = log_rs.reshape(-1, bsz)
        log_det = log_det.reshape(-1, bsz)

        new_xs = new_xs[:: self.thinning]
        log_rs = log_rs[:: self.thinning]
        log_det = log_det[:: self.thinning]

        return new_xs, log_rs

    def step_euler(self, position, velocity, force):
        with torch.no_grad():
            velocity = (
                (1 - self.gamma * self.dt) * velocity
                + force * self.dt / self.energy.mass
                + self.std * torch.randn_like(position, device=position.device)
            )
            position = position + velocity * self.dt

        position = position.requires_grad_(True)
        energy = self.energy.p.norm_energy(position.reshape(-1, self.mol_ndim)) / self.beta
        force = -torch.autograd.grad(energy.sum(), position)[0]
        log_r = -(energy * self.beta)
        return position.detach(), velocity.detach(), force.detach(), log_r.detach()

    def step_baoab(self, position, velocity, force):
        with torch.no_grad():
            velocity_half = velocity + (self.dt_half / self.mass) * force
            position_half = position + (self.dt_half) * velocity_half
            velocity_half = velocity_half * self.exp_mgdt + (
                self.std_baoab * torch.randn_like(position, device=position.device)
            )
            position = position_half + (self.dt_half) * velocity_half

        position = position.requires_grad_(True)
        energy = self.energy.p.norm_energy(position.reshape(-1, self.mol_ndim)) / self.beta
        force = -torch.autograd.grad(energy.sum(), position)[0]
        velocity = velocity_half + (self.dt_half / self.mass) * force
        log_r = -(energy * self.beta)
        return position.detach(), velocity.detach(), force.detach(), log_r.detach()
