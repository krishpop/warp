from typing import Tuple
import numpy as np
import warp as wp
import torch
from warp.envs import ObjectTask
from warp.envs import builder_utils as bu
from warp.envs.common import HandType, ObjectType, ActionType  # , run_env
from warp.envs.environment import RenderMode, run_env


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
        object_type: ObjectType = ObjectType.SPRAY_BOTTLE,
        stiffness=0.0,
        damping=0.5,
        rew_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.01, 0.6, 0.6),
        hand_start_orientation: Tuple = (-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),
    ):
        env_name = hand_type.name + "Env"
        self.action_type = action_type
        self.hand_start_position = hand_start_position
        self.hand_start_orientation = hand_start_orientation
        self.hand_type = hand_type
        super().__init__(
            num_envs,
            num_obs,
            num_act_dict[(hand_type, action_type)],
            episode_length,
            seed,
            no_grad,
            render,
            stochastic_init,
            device,
            render_mode,
            stage_path,
            object_type,
            stiffness,
            damping,
            rew_params,
        )

        # self.joint_type = joint_type

        # self.init_sim()
        self.setup_autograd_vars()
        if self.use_graph_capture:
            self.graph_capture_params["bwd_model"].joint_attach_ke = self.joint_attach_ke
            self.graph_capture_params["bwd_model"].joint_attach_kd = self.joint_attach_kd

        self.simulate_params["ag_return_body"] = True

    def _pre_step(self):
        self.extras["hand_joint_target_pos"] = self.actions.view(self.num_envs, -1)

    def _get_obs_dict(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        obs_dict = {}
        obs_dict["hand_joint_pos"] = joint_q[
            :, self.hand_joint_start : self.hand_joint_start + self.hand_num_joint_axis
        ]
        obs_dict["hand_joint_vel"] = joint_qd[
            :, self.hand_joint_start : self.hand_joint_start + self.hand_num_joint_axis
        ]
        obs_dict["object_joint_pos"] = joint_qd[
            :, self.object_joint_start : self.object_joint_start + self.object_num_joint_axis
        ]
        self.extras.update(obs_dict)
        return obs_dict

    def create_articulation(self, builder):
        if self.hand_type == HandType.ALLEGRO:
            bu.create_allegro_hand(
                builder,
                self.action_type,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
            )
        self.hand_joint_names = builder.joint_name
        self.hand_joint_start = builder.joint_axis_start
        self.joint_target_pos_indices = np.concatenate(
            [
                np.arange(self.hand_joint_start[i], self.hand_joint_start[i + 1])
                for i in range(len(self.hand_joint_start) - 1)
                if self.hand_joint_type[i] in supported_joint_types[self.action_type]
            ]
        )
        self.hand_joint_type = builder.joint_type
        self.hand_num_joint_axis = builder.joint_axis_count
        self.num_joint_q = len(builder.joint_q)
        self.num_joint_qd = len(builder.joint_qd)
        self.num_joint_axis = builder.joint_axis_count

        object_articulation_builder = wp.sim.ModelBuilder()
        super().create_articulation(object_articulation_builder)
        self.object_joint_names = object_articulation_builder.joint_name
        self.object_joint_start = (
            object_articulation_builder.joint_axis_start[0] + self.hand_joint_start + self.hand_num_joint_axis
        )
        self.object_joint_type = object_articulation_builder.joint_type
        self.object_num_joint_axis = object_articulation_builder.joint_axis_count

        self.asset_builders.append(object_articulation_builder)


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
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_obs", type=int, default=36)
    parser.add_argument("--episode_length", type=int, default=1000)
    args = parser.parse_args()

    rew_params = {
        "hand_joint_pos_err": (
            lambda x, y: torch.abs(x - y).sum(dim=-1),
            ["hand_joint_target_pos", "hand_joint_pos"],
            1.0,
        )
    }
    run_env(
        lambda: HandObjectTask(
            args.num_envs,
            args.num_obs,
            args.episode_length,
            action_type=ActionType[args.action_type.upper()],
            object_type=ObjectType[args.object_type.upper()],
            hand_type=HandType[args.hand_type.upper()],
            render=args.render,
            stiffness=1000.0,
            rew_params=rew_params,
        )
    )

    # run_env(lambda: HandEnv(5, 1, 1000, action_type=ActionType.POSITION, hand_type=HandType.ALLEGRO))
