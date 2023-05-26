from collections import OrderedDict
from torch.nn.modules import activation
from warp.envs import WarpEnv
from warp.envs import builder_utils as bu
from warp.envs.common import ObjectType, run_env
from warp.envs.environment import RenderMode

import warp as wp
import numpy as np
import torch


class ObjectEnv(WarpEnv):
    obs_keys = ["object_joint_pos", "object_joint_vel"]
    gravity = 0.0

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
        stiffness=0.0,
        damping=0.5,
        rew_params=None,
    ):
        env_name = object_type.name + "Env"
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

        self.object_type = object_type
        self.object_cls = bu.OBJ_MODELS[self.object_type]
        self.stiffness = stiffness
        self.damping = damping
        if rew_params is None:
            rew_params = {
                "action_penalty": (bu.action_penalty, ["actions"], 1e-3),
            }
        self.rew_params = rew_params

        self.init_sim()  # sets up renderer, model, etc.
        self.num_actions = self.num_joint_axis  # number of actions to set joint target
        self.setup_autograd_vars()
        if self.use_graph_capture:
            self.graph_capture_params["bwd_model"].joint_attach_ke = self.joint_attach_ke
            self.graph_capture_params["bwd_model"].joint_attach_kd = self.joint_attach_kd

        self.simulate_params["ag_return_body"] = self.ag_return_body

    def step(self, actions):
        self.actions = actions
        actions = actions.flatten()
        del self.extras
        self.extras = OrderedDict(actions=self.actions)
        self.model.joint_target.assign(wp.from_torch(actions))
        self.update()
        self.calculateObservations()
        self.calculateReward()
        self.progress_buf += 1
        self.reset_buf = self.reset_buf | (self.progress_buf >= self.episode_length)
        if self.visualize:
            self.render()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

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
        obs_dict = {"object_joint_pos": joint_q, "object_joint_vel": joint_qd}
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
        self.num_joint_q = len(builder.joint_q)
        self.num_joint_qd = len(builder.joint_qd)
        self.num_joint_axis = builder.joint_axis_count
        self.start_pos = self.object_model.base_pos
        self.start_ori = self.object_model.base_ori


if __name__ == "__main__":
    # parse command line arguments for object_type and turn it unto the ObjectType enum
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_type", type=str, default="spray_bottle")
    parser.add_argument("--stiffness", type=float, default=10.0)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()
    object_type = ObjectType[args.object_type.upper()]

    run_env(lambda: ObjectEnv(5, 1, 1, 1000, object_type=object_type, stiffness=args.stiffness, damping=args.damping))
