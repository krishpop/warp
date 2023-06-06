from collections import OrderedDict

import numpy as np
import torch
from torch.nn.modules import activation

import warp as wp

from .environment import RenderMode
from .utils.common import (
    ActionType,
    ObjectType,
    joint_coord_map,
    run_env,
    supported_joint_types,
)
from .utils import builder as bu
from .utils.rewards import action_penalty, l1_dist, parse_reward_params
from .utils.warp_utils import assign_act
from .utils.torch_utils import to_torch
from .warp_env import WarpEnv


class ObjectTask(WarpEnv):
    obs_keys = ["object_joint_pos", "object_joint_vel", "goal_joint_pos"]
    opengl_render_settings = {"draw_axis": False}
    # show_joints = True

    def __init__(
        self,
        num_envs,
        num_obs=1,
        num_act=None,
        action_type: ActionType = ActionType.TORQUE,
        episode_length: int = 200,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type: ObjectType = ObjectType.SPRAY_BOTTLE,
        object_id=0,
        stiffness=10.0,
        damping=1.0,
        reward_params=None,
        env_name=None,
        use_autograd=True,
        use_graph_capture=True,
        goal_joint_pos=None,
    ):
        if not env_name:
            if object_type:
                env_name = object_type.name + "Env"
            else:
                env_name = "ObjectEnv"
        self.num_joint_q = 0
        self.num_joint_qd = 0
        self.action_type = action_type
        if num_act is None:
            num_act = bu.get_num_acts(object_type)
        super().__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            seed,
            no_grad,
            render,
            stochastic_init,
            device,
            env_name,
            render_mode,
            stage_path,
        )

        self.use_autograd = use_autograd
        self.use_graph_capture = use_graph_capture

        self.object_type = object_type
        self.object_id = object_id
        self.action_type = ActionType.POSITION

        if self.object_type:
            obj_generator = bu.OBJ_MODELS[self.object_type]
        else:
            obj_generator = None
        try:
            if isinstance(obj_generator, dict):
                obj_generator = obj_generator[str(object_id)]
        except KeyError:
            raise ValueError(
                f"Object ID {object_id} not found for object type {object_type}, valid options are: {obj_generator.keys()}"
            )

        self.object_cls = obj_generator

        self.stiffness = stiffness
        self.damping = damping

        if reward_params is None:
            reward_params = {
                "action_penalty": (action_penalty, ["action"], 1e-3),
                "object_pos_err": (l1_dist, ["object_pos", "target_pos"], -1),
            }
        else:
            reward_params = parse_reward_params(reward_params)
        self.reward_params = reward_params
        self.reward_extras = {}

        self.init_sim()  # sets up renderer, model, etc.

        # initialize goal_joint_pos after creating model
        if self.object_type is not None:
            self.set_goal_joint_pos(goal_joint_pos)

        self.simulate_params["ag_return_body"] = self.ag_return_body

    def set_goal_joint_pos(self, goal_joint_pos):
        if goal_joint_pos is None:
            goal_joint_pos = np.ones((self.num_envs, self.object_num_joint_axis)) * 0.9  # controls joint 90% of the way
        else:
            assert (
                len(goal_joint_pos) == self.object_num_joint_axis
            ), "Goal joint pos must match number of object joints"
            goal_joint_pos = np.array(goal_joint_pos)

        joint_upper = self.model.joint_limit_upper.numpy().reshape(self.num_envs, -1)
        joint_lower = self.model.joint_limit_lower.numpy().reshape(self.num_envs, -1)
        goal_joint_pos = (
            goal_joint_pos
            * (joint_upper[:, self.object_joint_target_indices] - joint_lower[:, self.object_joint_target_indices])
            + joint_lower[:, self.object_joint_target_indices]
        )

        self.goal_joint_pos = to_torch(goal_joint_pos, device=self.device).view(1, -1).repeat(self.num_envs, 1)

    def init_sim(self):
        super().init_sim()
        if not hasattr(self, "num_actions") or self.num_actions is None:
            self.num_actions = self.env_num_joints  # number of actions that set joint target
        else:
            assert (
                self.env_num_joints == self.num_actions
            ), f"Number of actions {self.num_actions} must match number of joints, {self.env_num_joints}"
        self.warp_actions = wp.zeros(
            self.num_envs * self.num_actions, device=self._device, requires_grad=self.requires_grad
        )
        joint_axis_start = (
            self.model.joint_axis_start.numpy().reshape(self.num_envs, -1)[:, self.env_joint_mask].flatten()
        )
        joint_types = self.model.joint_type.numpy().reshape(self.num_envs, -1)[:, self.env_joint_mask].flatten()
        joint_target_indices = np.concatenate(
            [
                np.arange(joint_idx, joint_idx + joint_coord_map[joint_type])
                for joint_idx, joint_type in zip(joint_axis_start, joint_types)
            ]
        )
        self.joint_target_indices = wp.array(joint_target_indices, device=self._device, dtype=int)
        if not isinstance(self.stiffness, float):
            target_ke = self.model.joint_target_ke.numpy().reshape(self.num_envs, -1)
            target_kd = self.model.joint_target_kd.numpy().reshape(self.num_envs, -1)
            target_ke[:, self.object_joint_target_indices] = self.stiffness
            target_kd[:, self.object_joint_target_indices] = self.damping
            self.model.joint_target_ke.assign(target_ke.flatten())
            self.model.joint_target_kd.assign(target_kd.flatten())

        self.setup_autograd_vars()
        if self.use_graph_capture and self.use_autograd:
            self.graph_capture_params["bwd_model"].joint_attach_ke = self.joint_attach_ke
            self.graph_capture_params["bwd_model"].joint_attach_kd = self.joint_attach_kd

    def assign_actions(self, actions):
        self.warp_actions.zero_()
        self.warp_actions.assign(wp.from_torch(actions))
        self.model.joint_target.zero_()
        assign_act(
            self.warp_actions,
            self.model.joint_target,
            self.model.joint_target_ke,
            self.action_type,
            self.num_actions,
            self.num_envs,
            joint_indices=self.joint_target_indices,
        )
        # self.model.joint_target.assign(wp.from_torch(actions))

    def step(self, actions):
        self.actions = actions
        actions = actions.flatten()
        prev_extras = self.extras
        self.extras = OrderedDict(actions=self.actions)
        self.extras.update({f"prev_{k}": v for k, v in prev_extras.items()})
        self.assign_actions(actions)
        self._pre_step()
        if self.requires_grad and self.use_autograd:
            self.record_forward_simulate(actions)
        else:
            self.update()
        self._post_step()
        self.calculateObservations()
        self.calculateReward()
        self.progress_buf += 1
        self.num_frames += 1
        self.reset_buf = self.reset_buf | (self.progress_buf >= self.episode_length)
        if self.visualize:
            self.render()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _post_step(self):
        self.extras["body_f_max"] = self.body_f.max().item()

    def _get_rew_dict(self):
        rew_dict = {}
        for k, (cost_fn, rew_terms, rew_scale) in self.reward_params.items():
            rew_args = []
            for arg in rew_terms:
                if isinstance(arg, str) and arg in self.reward_extras:
                    rew_args.append(self.reward_extras[arg])
                elif isinstance(arg, str) and arg in self.extras:
                    rew_args.append(self.extras[arg])
                else:
                    raise TypeError("Invalid argument for reward function {}, ('{}')".format(k, arg))
            v = cost_fn(*rew_args) * rew_scale
            rew_dict[k] = v.view(self.num_envs)

        self.extras.update(rew_dict)
        return rew_dict

    def _get_obs_dict(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        obs_dict = {
            "object_joint_pos": joint_q,
            "object_joint_vel": joint_qd,
            "object_pos": joint_q[:, self.env_joint_target_indices],
            "target_pos": self.actions.view(self.num_envs, -1),
            "action": self.actions.view(self.num_envs, -1),
            "goal_joint_pos": self.goal_joint_pos.view(self.num_envs, -1),
        }
        self.extras.update(obs_dict)
        return obs_dict

    def calculateReward(self):
        rew_dict = self._get_rew_dict()
        self.rew_buf = torch.sum(torch.cat([v.view(self.num_envs, 1) for v in rew_dict.values()], dim=1), dim=-1).view(
            self.num_envs
        )
        return self.rew_buf

    def calculateObservations(self):
        obs_dict = self._get_obs_dict()
        self.obs_buf = obs_buf = torch.cat([obs_dict[k] for k in self.obs_keys], axis=1)
        return obs_buf

    def create_articulation(self, builder):
        if self.stiffness is None:
            self.object_model = self.object_cls()
        else:
            stiffness, damping = self.stiffness, self.damping
            self.object_model = self.object_cls(stiffness=stiffness, damping=damping)
        num_joints_before = len(builder.joint_type)
        num_joint_axis_before = builder.joint_axis_count
        self.object_model.create_articulation(builder)
        self.num_joint_q += builder.joint_axis_count - num_joint_axis_before
        self.num_joint_qd += builder.joint_axis_count - num_joint_axis_before

        valid_joint_types = supported_joint_types[self.action_type]
        self.env_joint_mask = [
            i + num_joints_before
            for i, joint_type in enumerate(builder.joint_type[num_joints_before:])
            if joint_type in valid_joint_types
        ]

        if len(self.env_joint_mask) > 0:
            joint_indices = []
            for i in self.env_joint_mask:
                joint_start, axis_count = builder.joint_axis_start[i], joint_coord_map[builder.joint_type[i]]
                joint_indices.append(np.arange(joint_start, joint_start + axis_count))
            joint_indices = np.concatenate(joint_indices)
        else:
            joint_indices = []

        self.env_num_joints = self.object_num_joint_axis = len(joint_indices)
        self.env_joint_target_indices = self.object_joint_target_indices = joint_indices
        if len(self.env_joint_mask) > 0:
            self.object_joint_start = self.env_joint_mask[0]
        else:
            self.object_joint_start = num_joint_axis_before

        self.start_pos = self.object_model.base_pos
        self.start_ori = self.object_model.base_ori


if __name__ == "__main__":
    # parse command line arguments for object_type and turn it unto the ObjectType enum
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_type", type=str, default="spray_bottle")
    parser.add_argument("--object_id", type=str, default=0)
    parser.add_argument("--stiffness", type=float, default=10.0)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    object_type = ObjectType[args.object_type.upper()]

    env = ObjectTask(
        5,
        13,
        bu.get_num_acts(object_type),
        1000,
        object_type=object_type,
        object_id=args.object_id,
        stiffness=args.stiffness,
        damping=args.damping,
        render=args.render,
    )
    run_env(env, log_runs=args.log)
