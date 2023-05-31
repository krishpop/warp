from collections import OrderedDict
from torch.nn.modules import activation
from warp.envs.warp_utils import assign_act
from warp.envs import WarpEnv
from warp.envs import builder_utils as bu
from warp.envs.common import ActionType, ObjectType, run_env, supported_joint_types, joint_coord_map
from warp.envs.environment import RenderMode

import warp as wp
import numpy as np
import torch


class ObjectTask(WarpEnv):
    obs_keys = ["object_joint_pos", "object_joint_vel"]

    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type: ObjectType = ObjectType.SPRAY_BOTTLE,
        object_id=0,
        stiffness=0.0,
        damping=0.5,
        rew_params=None,
        env_name=None,
        use_autograd=False,
    ):
        if not env_name:
            if object_type:
                env_name = object_type.name + "Env"
            else:
                env_name = "ObjectEnv"
        self.num_joint_q = 0
        self.num_joint_qd = 0
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

        self.object_type = object_type
        self.object_id = object_id
        self.action_type = ActionType.POSITION

        if self.object_type:
            obj_generator = bu.OBJ_MODELS[self.object_type]
        else:
            obj_generator = None
        if isinstance(obj_generator, dict):
            obj_generator = obj_generator[object_id]

        self.object_cls = obj_generator

        self.stiffness = stiffness
        self.damping = damping
        if rew_params is None:
            rew_params = {
                "action_penalty": (bu.action_penalty, ["actions"], 1e-3),
                "object_pos_err": (bu.abs_dist, ["object_pos", "target_pos"], -1),
            }
        self.rew_params = rew_params

        self.init_sim()  # sets up renderer, model, etc.
        self.simulate_params["ag_return_body"] = self.ag_return_body

    def init_sim(self):
        super().init_sim()
        self.num_actions = self.env_num_joints  # number of actions that set joint target
        self.warp_actions = wp.zeros(
            self.num_envs * self.num_actions, device=self.device, requires_grad=self.requires_grad
        )
        joint_axis_start = self.model.joint_axis_start.numpy()
        joint_type = self.model.joint_type.numpy()
        joint_target_indices = np.concatenate(
            [
                np.arange(joint_idx, joint_idx + joint_coord_map[joint_type])
                for joint_idx, joint_type in zip(joint_axis_start, joint_type)
                if joint_type in supported_joint_types[self.action_type]
            ],
        )
        self.joint_target_indices = wp.array(joint_target_indices, device=self.device, dtype=int)

        self.setup_autograd_vars()
        if self.use_graph_capture:
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
        del self.extras
        self.extras = OrderedDict(actions=self.actions)
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
        for k, (cost_fn, rew_terms, rew_scale) in self.rew_params.items():
            rew_args = [self.extras[k] for k in rew_terms]
            v = self.extras.get(k, cost_fn(*rew_args) * rew_scale)  # if pre-computed, use that
            assert np.prod(v.shape) == self.num_envs
            rew_dict[k] = v.view(self.num_envs, -1)
        return rew_dict

    def _get_obs_dict(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        obs_dict = {
            "object_joint_pos": joint_q,
            "object_joint_vel": joint_qd,
            "object_pos": joint_q[:, self.env_joint_target_indices],
            "target_pos": self.actions.view(self.num_envs, -1),
        }
        self.extras.update(obs_dict)
        return obs_dict

    def calculateReward(self):
        rew_dict = self._get_rew_dict()
        self.rew_buf = torch.sum(torch.cat([v for v in rew_dict.values()]), axis=-1, keepdim=True)
        self.extras.update(rew_dict)
        return self.rew_buf

    def calculateObservations(self):
        obs_dict = self._get_obs_dict()
        self.obs_buf = obs_buf = torch.cat([obs_dict[k] for k in self.obs_keys], axis=1)
        return obs_buf

    def create_articulation(self, builder):
        self.object_model = self.object_cls(stiffness=self.stiffness, damping=self.damping)
        self.object_model.create_articulation(builder)
        self.num_joint_q += len(builder.joint_q)
        self.num_joint_qd += len(builder.joint_qd)
        joint_indices = np.concatenate(
            [
                np.arange(joint_idx, joint_idx + joint_coord_map[joint_type])
                for joint_idx, joint_type in zip(builder.joint_axis_start, builder.joint_type)
                if joint_type in supported_joint_types[self.action_type]
            ]
        )

        self.env_num_joints = len(joint_indices)
        self.env_joint_target_indices = joint_indices

        self.start_pos = self.object_model.base_pos
        self.start_ori = self.object_model.base_ori


if __name__ == "__main__":
    # parse command line arguments for object_type and turn it unto the ObjectType enum
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_type", type=str, default="spray_bottle")
    parser.add_argument("--object_id", type=int, default=0)
    parser.add_argument("--stiffness", type=float, default=10.0)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()
    object_type = ObjectType[args.object_type.upper()]

    run_env(lambda: ObjectTask(5, 1, 1, 1000, object_type=object_type, stiffness=args.stiffness, damping=args.damping))
