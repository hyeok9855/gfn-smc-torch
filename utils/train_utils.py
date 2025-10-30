from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from models import GFN


def get_gfn_optimizer(
    gfn_model: "GFN",
    lr_fwd: float,
    lr_bwd: float,
    lr_flow: float,
    lr_logZ: float,
    lr_beta: float,
    lr_lgv: float,
    use_weight_decay=False,
    weight_decay=1e-7,
    use_scheduler=False,
    milestones: list[int] = [100000],
    gamma: float = 1.0,
):

    module_param_groups = gfn_model.pred_module.get_param_groups()

    param_groups = []
    param_groups.append({"params": module_param_groups.forward_params, "lr": lr_fwd})
    param_groups.append({"params": module_param_groups.backward_params, "lr": lr_bwd})
    param_groups.append({"params": module_param_groups.flow_params, "lr": lr_flow})
    param_groups.append({"params": module_param_groups.logZ_params, "lr": lr_logZ})
    param_groups.append({"params": module_param_groups.lgv_params, "lr": lr_lgv})

    if gfn_model.beta_model is not None:
        param_groups.append({"params": gfn_model.beta_model, "lr": lr_beta})

    gfn_optimizer = torch.optim.Adam(
        param_groups, lr=0.0, weight_decay=weight_decay if use_weight_decay else 0.0
    )

    gfn_scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(gfn_optimizer, milestones=milestones, gamma=gamma)
        if use_scheduler
        else None
    )
    return gfn_optimizer, gfn_scheduler


###########################################
### Importance weight related functions ###
###########################################


def solve_mixing_ratio(normalized_weights: torch.Tensor, target_ess: float) -> float:
    """
    Find the mixing ratio to achieve the target effective sample size (ESS)

    normalized_weights_mix = (1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size

    ESS_mix = 1 / (normalized_weights_mix^2).sum()
            = 1 / (((1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size)^2).sum()

    Solve the following equation for mixing_ratio:
    1 / ESS_mix = (((1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size)^2).sum()
                = 1 / target_ess

    This is equivalent to the following quadratic equation:
        A * (mixing_ratio^2) - 2 * B * mixing_ratio + C = 0
    where
        A = normalized_weights^2.sum() - 2 * normalized_weights.sum() / N + 1 / N
        B = normalized_weights^2.sum() - normalized_weights.sum() / N
        C = normalized_weights^2.sum() - 1 / target_ess
    """
    N = len(normalized_weights)
    nw_sum = 1.0
    nw_squared_sum = (normalized_weights**2).sum().item()

    ess_before = 1 / nw_squared_sum
    if ess_before >= target_ess:
        return 0.0

    A = nw_squared_sum - 2 * nw_sum / N + 1 / N
    B = nw_squared_sum - nw_sum / N
    C = nw_squared_sum - 1 / target_ess

    min_lhs = C - B**2 / A
    if min_lhs >= 0:
        raise ValueError(f"Cannot achieve target ESS: {target_ess}")
        return 1.0

    mixing_ratio_1 = (B + (B**2 - A * C) ** 0.5) / A
    mixing_ratio_2 = (B - (B**2 - A * C) ** 0.5) / A

    valid_1 = mixing_ratio_1 >= 0.0 and mixing_ratio_1 <= 1.0
    valid_2 = mixing_ratio_2 >= 0.0 and mixing_ratio_2 <= 1.0

    if valid_1 and valid_2:
        raise ValueError(f"Multiple solutions: {mixing_ratio_1}, {mixing_ratio_2}")
    elif valid_1:
        return mixing_ratio_1
    elif valid_2:
        return mixing_ratio_2
    else:
        raise ValueError(f"No valid solution: {mixing_ratio_1}, {mixing_ratio_2}")


def ess(
    log_weights: torch.Tensor | None = None,  # (bs, T)
    normalized_weights: torch.Tensor | None = None,  # (bs, T)
) -> torch.Tensor:
    if normalized_weights is None:
        assert log_weights is not None
        normalized_weights = log_weights.softmax(dim=0)  # (bs, T)
    return 1 / (normalized_weights**2).sum(dim=0)  # (T,)


def binary_search_smoothing(
    log_weights: torch.Tensor,  # (bs, T)
    target_ess: float,
    tol=1e-3,
    max_steps=1000,
) -> tuple[torch.Tensor, torch.Tensor]:  # (bs, T), (1, T)
    bs = log_weights.shape[0]

    search_min, search_max = get_min_max(log_weights)
    search_min = torch.tensor(search_min, device=log_weights.device).repeat(1, log_weights.shape[1])
    search_max = torch.tensor(search_max, device=log_weights.device).repeat(1, log_weights.shape[1])
    mid = (search_min + search_max) / 2  # (1, T)
    original_order = ess(log_weights / search_min) < ess(log_weights / search_max)

    dones = ess(log_weights=log_weights) / bs >= target_ess  # (T,)
    log_weights_smoothed = log_weights.clone()  # (bs, T)

    steps = 0
    while not dones.all():
        steps += 1
        mid[0, ~dones] = (search_min[0, ~dones] + search_max[0, ~dones]) / 2  # (1, T)

        new_log_weights = log_weights / mid  # (bs, T)
        new_ess = ess(log_weights=new_log_weights) / bs  # (T,)
        new_dones = (~dones) & (abs(new_ess - target_ess) / target_ess < tol)  # (T,)
        log_weights_smoothed[:, new_dones] = new_log_weights[:, new_dones]
        dones = dones | new_dones

        search_max = torch.where((new_ess > target_ess) == original_order, mid, search_max)
        search_min = torch.where((new_ess < target_ess) == original_order, mid, search_min)

        if steps > max_steps:
            print(f"Warning: Binary search failed in {max_steps} steps")
            log_weights_smoothed[:, ~dones] = new_log_weights[:, ~dones]
            break
    return log_weights_smoothed, mid


def get_min_max(log_weights: torch.Tensor) -> tuple[float, float]:
    _min = torch.nan_to_num(log_weights, nan=float("inf"), neginf=float("inf")).min().item()
    _max = torch.nan_to_num(log_weights, nan=float("-inf"), posinf=float("-inf")).max().item()
    return 1.0, (_max - _min) / 2
