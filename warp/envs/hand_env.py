from typing import Tuple
import numpy as np
from warp.envs import ObjectEnv
from warp.envs import builder_utils as bu
from warp.envs.common import HandType, ObjectType, ActionType, run_env
from warp.envs.environment import RenderMode


num_act_dict = {
    (HandType.ALLEGRO, ActionType.POSITION): 16,
    (HandType.ALLEGRO, ActionType.TORQUE): 16,
    (HandType.ALLEGRO, ActionType.VARIABLE_STIFFNESS): 32,
    (HandType.SHADOW, ActionType.POSITION): 24,
    (HandType.SHADOW, ActionType.TORQUE): 24,
}


class HandEnv(ObjectEnv):
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
        hand_start_position: Tuple = (0.01, 0.6, 0.125),
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

    def create_articulation(self, builder):
        if self.hand_type == HandType.ALLEGRO:
            bu.create_allegro_hand(
                builder,
                self.action_type,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
            )
        super().create_articulation(builder)
        # self.num_joint_q = len(builder.joint_q)
        # self.num_joint_qd = len(builder.joint_qd)
        # self.num_joint_axis = builder.joint_axis_count


if __name__ == "__main__":
    # operable_object_generator(
    #     ObjectType.SPRAY_BOTTLE,
    #     base_pos=(0.0, 0.22, 0.0),
    #     base_ori=(0.0, 0.0, 0.0),
    #     scale=1.0,
    #     model_path="spray_bottle/mobility.urdf",
    # )

    run_env(lambda: HandEnv(5, 1, 1000, action_type=ActionType.POSITION, hand_type=HandType.ALLEGRO))
