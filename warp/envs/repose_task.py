import numpy as np
import torch
from warp.envs.common import (
    HandType,
    ObjectType, ActionType,
    run_env,
)
from warp.envs.environment import RenderMode
from warp.envs.hand_env import HandObjectTask
from warp.envs.rewards import action_penalty, l2_dist, rot_dist, rot_reward, reach_bonus

from typing import Tuple


class ReposeTask(HandObjectTask):
    obs_keys = ["hand_joint_pos", "hand_joint_vel", "object_pos", "target_pos"]

    def __init__(
        self,
        num_envs,
        num_obs,
        episode_length,
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
        hand_start_position: Tuple = (0.0, 0.3, -0.6),
        hand_start_orientation: Tuple = (-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),
        use_autograd: bool = False,
        reach_threshold=0.1,
    ):
        object_type = ObjectType.REPOSE_CUBE
        object_id = 0
        rew_dict_params = {k: v for k, v in rew_params.items() if k != "reach_bonus"}
        self.reach_bonus = rew_params.get("reach_bonus", (lambda x: torch.ones_like(x), ("object_pose_err"), 100.0))
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
            rew_params=rew_dict_params,
            hand_type=hand_type,
            hand_start_position=hand_start_position,
            hand_start_orientation=hand_start_orientation,
            grasps=None,
            use_autograd=use_autograd,
        )
        self.rew_extras["reach_threshold"] = reach_threshold
        self.goal_pos = torch.as_tensor([0.0, 0.3, -0.6], dtype=float, device=self.device)

    def _get_object_pose(self):
        joint_q = self.joint_q.view(self.num_envs, -1)
        object_joint_pos = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        pose = {}
        pose["position"] = object_joint_pos
        return object_joint_pos

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["target_pos"] = self.goal_pos
        object_pose = self._get_object_pose()
        obs_dict["object_pos"] = object_pose["position"]
        # obs_dict["object_rot"] = object_pose['orientation']
        obs_dict["object_pose_err"] = l2_dist(obs_dict["object_pos"], obs_dict["target_pos"])
        # obj_dict["object_pose_err"] += rot_dist(self.object_rot, self.target_rot)
        return obs_dict

    def _get_rew_dict(self):
        rew_dict = super()._get_rew_dict()
        cost_fn, rew_args, rew_scale = self.reach_bonus
        rew_args = [rew_dict[k] for k in rew_args]
        rew_dict["reach_bonus"] = cost_fn(*rew_args) * rew_scale
        return rew_dict


if __name__ == "__main__":
    reach_bonus = lambda x: torch.where(x < 0.1, torch.ones_like(x), torch.zeros_like(x))
    rew_params = {
        "object_pos_err": (l2_dist, ("target_pos", "object_pos"), -10.0),
        "rot_reward": (rot_reward, ("object_rot", "target_rot"), 1.0),
        "action_penalty": (action_penalty, ("action"), -0.0002),
        "reach_bonus": (reach_bonus, ("object_pose_err", "reach_threshold"), 250.0),
    }

    run_env(lambda: ReposeTask(num_envs=1, num_obs=1, episode_length=1000, rew_params=rew_params))
