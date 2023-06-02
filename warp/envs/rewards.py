import torch
from warp.envs.torch_utils import quat_conjugate, quat_mul

action_penalty = lambda act: torch.linalg.norm(act, dim=-1)
l2_dist = lambda x, y: torch.linalg.norm(x - y, dim=-1)
l1_dist = lambda x, y: torch.abs(x - y).sum(dim=-1)


@torch.jit.script
def rot_dist(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
    return 2.0 * rot_dist


@torch.jit.script
def rot_reward(object_rot, target_rot):
    return 1.0 / torch.abs(rot_dist(object_rot, target_rot) + 0.1)


@torch.jit.script
def reach_bonus(pose_err, threshold: float = 0.1):
    return torch.where(pose_err < threshold, torch.ones_like(pose_err), torch.zeros_like(pose_err))
