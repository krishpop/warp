from typing import Tuple, Optional, List
import numpy as np
import warp as wp
import torch
from warp.envs import ObjectTask
from warp.envs import builder_utils as bu
from warp.envs.common import (
    HandType,
    ObjectType,
    ActionType,
    GraspParams,
    joint_coord_map,
    supported_joint_types,
    run_env,
)
from warp.envs.environment import RenderMode


num_act_dict = {
    (HandType.ALLEGRO, ActionType.POSITION): 16,
    (HandType.ALLEGRO, ActionType.TORQUE): 16,
    (HandType.ALLEGRO, ActionType.VARIABLE_STIFFNESS): 32,
    (HandType.SHADOW, ActionType.POSITION): 24,
    (HandType.SHADOW, ActionType.TORQUE): 24,
}


class HandObjectTask(ObjectTask):
    obs_keys = ["hand_joint_pos", "hand_joint_vel"]

    def __init__(
        self,
        num_envs,
        num_obs,
        episode_length,
        action_type: ActionType = ActionType.TORQUE,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type: Optional[ObjectType] = None,
        object_id=0,
        stiffness=0.0,
        damping=0.5,
        rew_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.0, 0.3, -0.6),
        hand_start_orientation: Tuple = (-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),
        grasps: List[GraspParams] = None,
        use_autograd: bool = False,
    ):
        env_name = hand_type.name + "Env"
        self.action_type = action_type
        self.hand_start_position = hand_start_position
        self.hand_start_orientation = hand_start_orientation
        self.hand_type = hand_type
        self.grasps = grasps
        stochastic_init = stochastic_init or (grasps is not None)
        self.hand_stiffness = stiffness
        self.hand_damping = damping
        # self.gravity = 0.0

        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act_dict[(hand_type, action_type)],
            episode_length=episode_length,
            seed=seed,
            no_grad=no_grad,
            render=render,
            stochastic_init=stochastic_init,
            device=device,
            render_mode=render_mode,
            stage_path=stage_path,
            object_type=object_type,
            object_id=object_id,
            stiffness=0.0,
            damping=damping,
            rew_params=rew_params,
            env_name=env_name,
            use_autograd=use_autograd,
        )

        print("gravity", self.model.gravity, self.gravity)
        self.hand_target_ke = self.model.joint_target_ke
        self.hand_target_kd = self.model.joint_target_kd

        self.setup_autograd_vars()

        self.simulate_params["ag_return_body"] = True

    def _post_step(self):
        self.extras["target_qpos"] = self.actions.view(self.num_envs, -1)
        self.extras["hand_qpos"] = self.joint_q.view(self.num_envs, -1)[:, self.env_joint_target_indices]
        self.extras["body_f_max"] = self.body_f.max().item()

    def _get_obs_dict(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        obs_dict = {}
        obs_dict["hand_joint_pos"] = joint_q[
            :, self.hand_joint_start : self.hand_joint_start + self.hand_num_joint_axis
        ]
        obs_dict["hand_joint_vel"] = joint_qd[
            :, self.hand_joint_start : self.hand_joint_start + self.hand_num_joint_axis
        ]
        if self.object_type is not None:
            obs_dict["object_joint_pos"] = joint_q[
                :, self.object_joint_start : self.object_joint_start + self.object_num_joint_axis
            ]
            obs_dict["object_joint_vel"] = joint_qd[
                :, self.object_joint_start : self.object_joint_start + self.object_num_joint_axis
            ]
        self.extras.update(obs_dict)
        return obs_dict

    def sample_grasps(self, num_envs):
        self.grasp = self.grasps[np.random.randint(len(self.grasps), size=num_envs)]
        self.hand_init_xform = np.stack([g.xform for g in self.grasp], axis=0)
        self.hand_init_q = np.stack([g.q for g in self.grasp], axis=0)

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        # need to set the base joint of each env to sampled grasp xform
        # then set each joint target pos to grasp.
        joint_q, joint_qd = super().get_stochastic_init()
        if self.grasps is not None:
            assert joint_q.shape[-1] == self.hand_init_q.shape[-1]
            joint_q[env_ids, self.env_joint_target_indices] = self.hand_init_q.copy()

        return joint_q, joint_qd

    def reset(self, env_ids=None, force_reset=True):
        if self.grasps is not None:
            self.sample_grasps()
        super().reset(env_ids=env_ids, force_reset=force_reset)

    def create_articulation(self, builder):
        if self.hand_type == HandType.ALLEGRO:
            bu.create_allegro_hand(
                builder,
                self.action_type,
                stiffness=self.hand_stiffness,
                damping=self.hand_damping,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
            )
        elif self.hand_type == HandType.SHADOW:
            bu.create_shadow_hand(
                builder,
                self.action_type,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
            )
        else:
            raise NotImplementedError("Hand type not supported:", self.hand_type)

        self.hand_joint_names = builder.joint_name
        if self.actions.max() > 0.1:
            __import__("ipdb").set_trace()
        self.hand_joint_start = builder.joint_axis_start[0]
        joint_indices = np.concatenate(
            [
                np.arange(joint_idx, joint_idx + joint_coord_map[joint_type])
                for joint_idx, joint_type in zip(builder.joint_axis_start, builder.joint_type)
                if joint_type in supported_joint_types[self.action_type]
            ]
        )
        self.hand_num_joint_axis = builder.joint_axis_count
        self.num_joint_q += len(builder.joint_q)
        self.num_joint_qd += len(builder.joint_qd)

        if self.object_type:
            object_articulation_builder = wp.sim.ModelBuilder()
            super().create_articulation(object_articulation_builder)
            self.object_num_joint_axis = object_articulation_builder.joint_axis_count
            self.object_joint_start = (
                object_articulation_builder.joint_axis_start[0] + self.hand_joint_start + self.hand_num_joint_axis
            )
            self.object_joint_type = object_articulation_builder.joint_type
            self.object_num_joint_axis = object_articulation_builder.joint_axis_count
            self.asset_builders.insert(0, object_articulation_builder)

        self.env_joint_target_indices = joint_indices
        self.env_num_joints = len(joint_indices)


if __name__ == "__main__":
    # operable_object_generator(
    #     ObjectType.SPRAY_BOTTLE,
    #     base_pos=(0.0, 0.22, 0.0),
    #     base_ori=(0.0, 0.0, 0.0),
    #     scale=1.0,
    #     model_path="spray_bottle/mobility.urdf",
    # )
    # create argparse params for handenv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_type", type=str, default="allegro")
    parser.add_argument("--action_type", type=str, default="position")
    parser.add_argument("--object_type", type=str, default="spray_bottle")
    parser.add_argument("--object_id", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--norender", action="store_false", dest="render")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_obs", type=int, default=36)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--stiffness", type=float, default=5000.0)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--profile", action="store_true")
    parser.set_defaults(render=True)

    args = parser.parse_args()

    rew_params = {
        "hand_joint_pos_err": (
            bu.l1_dist,
            ["target_qpos", "hand_qpos"],
            1.0,
        )
    }
    HandObjectTask.profile = args.profile
    run_env(
        lambda: HandObjectTask(
            args.num_envs,
            args.num_obs,
            args.episode_length,
            action_type=ActionType[args.action_type.upper()],
            object_type=ObjectType[args.object_type.upper()],
            object_id=args.object_id,
            hand_type=HandType[args.hand_type.upper()],
            render=args.render,
            stiffness=args.stiffness,
            damping=args.damping,
            rew_params=rew_params,
        )
    )
