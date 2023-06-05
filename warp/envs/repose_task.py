from typing import Tuple

import numpy as np
import torch

from .utils import torch_utils as tu
from .environment import RenderMode
from .hand_env import HandObjectTask
from .utils.common import ActionType, HandType, ObjectType, run_env
from .utils.rewards import action_penalty, l2_dist, reach_bonus, rot_dist, rot_reward


class ReposeTask(HandObjectTask):
    obs_keys = ["hand_joint_pos", "hand_joint_vel", "object_pos", "target_pos"]
    debug_visualization = False

    def __init__(
        self,
        num_envs,
        num_obs=1,
        episode_length=500,
        action_type: ActionType = ActionType.POSITION,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        stiffness=0.0,
        damping=0.5,
        rew_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.1, 0.3, 0.0),
        hand_start_orientation: Tuple = (-np.pi / 2, np.pi * 0.75, np.pi / 2),
        use_autograd: bool = False,
        reach_threshold: float = 0.1,
        reach_bonus: float = 100.0,
    ):
        object_type = ObjectType.REPOSE_CUBE
        object_id = 0
        reward_params_dict = {k: v for k, v in rew_params.items() if k != "reach_bonus"}
        if "reach_bonus" not in rew_params:
            rew_params["reach_bonus"] = (lambda x: torch.zeros_like(x), ("object_pose_err",), reach_bonus)
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            episode_length=episode_length,
            action_type=action_type,
            seed=seed,
            no_grad=no_grad,
            render=render,
            stochastic_init=stochastic_init,
            device=device,
            render_mode=render_mode,
            stage_path=stage_path,
            object_type=object_type,
            object_id=object_id,
            stiffness=stiffness,
            damping=damping,
            reward_params=reward_params_dict,
            hand_type=hand_type,
            hand_start_position=hand_start_position,
            hand_start_orientation=hand_start_orientation,
            grasp_file="",
            grasp_id=None,
            use_autograd=use_autograd,
        )
        self.reward_extras["reach_threshold"] = reach_threshold
        # stay in center of hand
        self.goal_pos = torch.as_tensor([0.0, 0.32, 0.0], dtype=float, device=str(self.device))
        self.goal_rot = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=float, device=str(self.device))

    def _get_object_pose(self):
        joint_q = self.joint_q.view(self.num_envs, -1)

        pose = {}
        if self.object_model.floating:
            object_joint_pos = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
            object_joint_quat = joint_q[:, self.object_joint_start + 3 : self.object_joint_start + 7]
            pose["position"] = object_joint_pos + tu.to_torch(self.object_model.base_pos).view(1, 3)
            pose["orientation"] = tu.quat_mul(object_joint_quat, tu.to_torch(self.object_model.base_quat).view(1, 4))
        elif self.object_model.base_joint == "px, py, px":
            pose["position"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        elif self.object_model.base_joint == "rx, ry, rx":
            pose["orientation"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        elif self.object_model.base_joint == "px, py, pz, rx, ry, rz":
            pose["position"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
            pose["orientation"] = joint_q[:, self.object_joint_start + 3 : self.object_joint_start + 6]
        return pose

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["target_pos"] = self.goal_pos.view(1, 3).repeat(self.num_envs, 1)
        obs_dict["target_quat"] = self.goal_rot.view(1, 4).repeat(self.num_envs, 1)
        object_pose = self._get_object_pose()
        obs_dict["object_pos"] = object_pose["position"]
        # obs_dict["object_rot"] = object_pose['orientation']
        obs_dict["object_pose_err"] = l2_dist(obs_dict["object_pos"], obs_dict["target_pos"]).view(self.num_envs)
        # obj_dict["object_pose_err"] += rot_dist(
        #     object_pos["orientation"], self.goal_rot.view(1, 3).repeat(self.num_envs, 1)
        # )
        obs_dict["action"] = self.actions.view(self.num_envs, -1)
        self.extras.update(obs_dict)
        return obs_dict

    def _get_rew_dict(self):
        rew_dict = super()._get_rew_dict()
        # cost_fn, rew_args, rew_scale = self.reach_bonus
        # args = []
        # for arg in rew_args:
        #     if arg in self.extras:
        #         args.append(self.extras[arg])
        #     elif arg in rew_dict:
        #         args.append(rew_dict[arg])
        #
        # rew_dict["reach_bonus"] = cost_fn(*args) * rew_scale
        return rew_dict

    def render(self, **kwargs):
        super().render(**kwargs)
        if self.debug_visualization:
            points = [self.extras["object_pos"]]
            self.renderer.render_points("debug_markers", points, radius=0.015)


if __name__ == "__main__":
    reach_bonus = lambda x, y: torch.where(x < y, torch.ones_like(x), torch.zeros_like(x))
    rew_params = {
        "object_pos_err": (l2_dist, ("target_pos", "object_pos"), -10.0),
        # "rot_reward": (rot_reward, ("object_rot", "target_rot"), 1.0),
        "action_penalty": (action_penalty, ("action",), -0.0002),
        "reach_bonus": (reach_bonus, ("object_pose_err", "reach_threshold"), 250.0),
    }

    run_env(ReposeTask(num_envs=1, num_obs=38, episode_length=1000, rew_params=rew_params), pi=None)
