import torch
import warp as wp
from omegaconf import DictConfig
from hydra.utils import instantiate
from .torch_utils import quat_conjugate, quat_mul
from .warp_utils import quat_ang_err, compute_ctrl_reward

action_penalty = lambda act: torch.linalg.norm(act, dim=-1)
action_penalty_kernel = compute_ctrl_reward
l2_dist = lambda x, y: torch.linalg.norm(x - y, dim=-1)
l1_dist = lambda x, y: torch.abs(x - y).sum(dim=-1)


@torch.jit.script
def l2_dist_exp(x, y, eps: float = 1e-1):
    return torch.exp(-torch.linalg.norm(x - y, dim=-1) / eps)


@torch.jit.script
def rot_dist(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
    return 2.0 * rot_dist


@torch.jit.script
def rot_reward(object_rot, target_rot, rot_eps: float = 0.1):
    return 1.0 / torch.abs(rot_dist(object_rot, target_rot) + rot_eps)


@torch.jit.script
def rot_dist_delta(object_rot, target_rot, prev_rot_dist):
    return prev_rot_dist - rot_dist(object_rot, target_rot)


@torch.jit.script
def reach_bonus(pose_err, threshold: float = 0.1):
    return torch.where(pose_err < threshold, torch.ones_like(pose_err), torch.zeros_like(pose_err))


def parse_reward_params(reward_params):
    rew_params = {}
    if isinstance(reward_params, list):
        for value in reward_params:
            value.rstrip("_kernel")
            if value == "action_penalty":
                rew_params[value] = action_penalty_kernel
            # elif value == "l2_dist":
            #     rew_params[value] = l2_dist_kernel
            # elif value == "l1_dist":
            #     rew_params[value] = l1_dist_kernel
            # elif value == "object_pos_err":
            #     rew_params[value] = pos_dist_kernel
            # elif value == "l2_dist_exp":
            #     rew_params[value] = l2_dist_exp_kernel
            # elif value == "rot_dist":
            #     rew_params[value] = rot_dist_kernel
            # elif value == "rot_reward":
            #     rew_params[value] = rot_reward_kernel

    else:
        for key, value in reward_params.items():
            if isinstance(value, (DictConfig, dict)):
                function = value["reward_fn_partial"]
                arguments = value["args"]
                if isinstance(arguments, str):
                    arguments = [arguments]
                coefficient = value["scale"]
            else:
                function, arguments, coefficient = value
            rew_params[key] = (function, arguments, coefficient)
    return rew_params
