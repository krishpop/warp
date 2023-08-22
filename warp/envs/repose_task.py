from typing import Tuple

import numpy as np
import torch

from .utils import torch_utils as tu
from .environment import RenderMode
from .hand_env import HandObjectTask
from .utils.common import ActionType, HandType, ObjectType, run_env, profile
from .utils.rewards import action_penalty, l2_dist, reach_bonus, rot_dist, rot_reward


class ReposeTask(HandObjectTask):
    obs_keys = ["hand_joint_pos", "hand_joint_vel", "object_pos", "target_pos"]
    debug_visualization = False
    drop_height: float = 0.23

    def __init__(
        self,
        num_envs,
        num_obs=38,
        episode_length=500,
        action_type: ActionType = ActionType.POSITION,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        logdir=None,
        stiffness=5000.0,
        damping=10.0,
        reward_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.1, 0.3, 0.0),
        hand_start_orientation: Tuple = (-np.pi / 2, np.pi * 0.75, np.pi / 2),
        use_autograd: bool = True,
        use_graph_capture: bool = True,
        reach_threshold: float = 0.1,
        headless=False,
    ):
        stage_path = stage_path or logdir
        object_type = ObjectType.REPOSE_CUBE
        object_id = 0
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
            reward_params=reward_params,
            hand_type=hand_type,
            hand_start_position=hand_start_position,
            hand_start_orientation=hand_start_orientation,
            load_grasps=False,
            grasp_id=None,
            use_autograd=use_autograd,
            use_graph_capture=use_graph_capture,
            headless=headless,
        )
        self.reward_extras["reach_threshold"] = reach_threshold
        # stay in center of hand
        self.goal_pos = self.default_goal_pos = (
            tu.to_torch([0.0, 0.32, 0.0], device=self.device).view(1, 3).repeat(self.num_envs, 1)
        )
        self.goal_rot = self.default_goal_rot = (
            tu.to_torch([0.0, 0.0, 0.0, 1.0], device=self.device).view(1, 4).repeat(self.num_envs, 1)
        )

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        goal_pos = self.goal_pos[env_ids]
        sample_rot = tu.torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        x, _, z = torch.eye(3, device=self.device)
        goal_rot_x = tu.quat_from_angle_axis(sample_rot[:, 0], x)
        goal_rot_z = tu.quat_from_angle_axis(sample_rot[:, 0], z)
        self.goal_rot[env_ids] = tu.quat_mul(goal_rot_x, goal_rot_z)
        return joint_q[env_ids], joint_qd[env_ids]

    def _get_object_pose(self):
        joint_q = self.joint_q.view(self.num_envs, -1)

        pose = {}
        if self.object_model.floating:
            object_joint_pos = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
            object_joint_quat = joint_q[:, self.object_joint_start + 3 : self.object_joint_start + 7]
            if self.use_tiled_rendering:
                pose["position"] = object_joint_pos  #  + tu.to_torch(self.object_model.base_pos).view(1, 3)
            else:
                pose["position"] = object_joint_pos - self.env_offsets
            start_quat = tu.to_torch(self.object_model.base_ori).view(1, 4).repeat(self.num_envs, 1)
            pose["orientation"] = tu.quat_mul(object_joint_quat, start_quat)
        elif self.object_model.base_joint == "px, py, px":
            pose["position"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        elif self.object_model.base_joint == "rx, ry, rx":
            pose["orientation"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        elif self.object_model.base_joint == "px, py, pz, rx, ry, rz":
            pose["position"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
            pose["orientation"] = joint_q[:, self.object_joint_start + 3 : self.object_joint_start + 6]
        return pose

    def _pre_step(self):
        rot_dist = torch.zeros_like(self.rew_buf) if self._prev_obs is None else self._prev_obs["rot_dist"]
        self.reward_extras["prev_rot_dist"] = torch.where(
            self.progress_buf == 0,
            torch.zeros_like(rot_dist),
            rot_dist,
        )

    def _check_early_termination(self, obs_dict):
        # check if object is dropped
        object_body_pos = obs_dict["object_pos"]
        termination = object_body_pos[:, 1] < self.drop_height
        self.termination_buf = self.termination_buf | termination
        self.reset_buf = self.reset_buf | termination
        return termination

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["target_pos"] = self.goal_pos
        obs_dict["target_rot"] = self.goal_rot
        object_pose = self._get_object_pose()
        obs_dict["object_pos"] = object_pose["position"]
        obs_dict["object_rot"] = object_pose["orientation"]
        obs_dict["object_pose_err"] = l2_dist(obs_dict["object_pos"], obs_dict["target_pos"]).view(self.num_envs, 1)
        obs_dict["rot_dist"] = rot_dist(object_pose["orientation"], obs_dict["target_rot"])

        self.extras["obs_dict"] = obs_dict

        # log score keys
        self.extras["object_pose_err"] = obs_dict["object_pose_err"].view(self.num_envs)
        self.extras["object_rot_err"] = obs_dict["rot_dist"].view(self.num_envs)
        if self.action_type is ActionType.TORQUE:
            self.extras["net_energy"] = torch.bmm(
                obs_dict["hand_joint_vel"].unsqueeze(1), self.actions.unsqueeze(2)
            ).squeeze()
        else:
            self.extras["net_energy"] = torch.zeros_like(self.rew_buf)
        termination = self._check_early_termination(obs_dict)
        self.extras["termination"] = termination
        return obs_dict

    def render(self, **kwargs):
        super().render(**kwargs)
        if self.debug_visualization:
            points = [self.extras["object_pos"]]
            self.renderer.render_points("debug_markers", points, radius=0.015)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", "-ne", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--norender", action="store_true")
    args = parser.parse_args()

    reach_bonus = lambda x, y: torch.where(x < y, torch.ones_like(x), torch.zeros_like(x))
    reward_params = {
        "object_pos_err": (l2_dist, ("target_pos", "object_pos"), -10.0),
        # "rot_reward": (rot_reward, ("object_rot", "target_rot"), 1.0),
        "action_penalty": (action_penalty, ("action",), -0.0002),
        "reach_bonus": (reach_bonus, ("object_pose_err", "reach_threshold"), 250.0),
    }
    if args.profile or args.norender:
        render_mode = RenderMode.NONE
    else:
        render_mode = RenderMode.OPENGL
    env = ReposeTask(
        num_envs=args.num_envs, num_obs=38, episode_length=1000, reward_params=reward_params, render_mode=render_mode
    )
    if args.profile:
        profile(env)
    else:
        env.load_camera_params()
        run_env(env, pi=None, num_rollouts=args.num_rollouts, logdir="outputs/")
