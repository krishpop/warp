import os
import numpy as np
import warp as wp
from typing import Tuple, Optional
from .obj_env import ObjectTask
from .utils import builder as bu
from .utils.rewards import l1_dist
from .utils.common import (
    HandType,
    ObjectType,
    ActionType,
    HAND_ACT_COUNT,
    joint_coord_map,
    supported_joint_types,
    run_env,
)
from .environment import RenderMode


class HandObjectTask(ObjectTask):
    obs_keys = ["hand_joint_pos", "hand_joint_vel"]
    fix_position: bool = True
    fix_orientation: bool = True
    collapse_joints: bool = True

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
        stiffness=1000.0,
        damping=0.5,
        reward_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.1, 0.3, 0.0),
        hand_start_orientation: Tuple = (-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),
        grasp_file: str = "",
        grasp_id: int = None,
        use_autograd: bool = True,
        use_graph_capture: bool = True,
        goal_joint_pos=None,
    ):
        env_name = hand_type.name + "Env"
        self.hand_start_position = hand_start_position
        self.hand_start_orientation = hand_start_orientation
        self.hand_type = hand_type

        if os.path.exists(grasp_file):
            all_grasps = np.load(grasp_file, allow_pickle=False)
            if grasp_id != None:
                self.grasps = all_grasps[grasp_id]
            else:
                self.grasps = all_grasps[np.random.randint(0, all_grasps.shape[0])]
        else:
            self.grasps = None
            if grasp_file != "":
                print(f"Grasp file {grasp_file} not found")

        stochastic_init = stochastic_init or (self.grasps is not None)
        self.hand_stiffness = stiffness
        self.hand_damping = damping
        # self.gravity = 0.0

        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=HAND_ACT_COUNT[(hand_type, action_type)],
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
            stiffness=0.0,
            damping=damping,
            reward_params=reward_params,
            env_name=env_name,
            use_autograd=use_autograd,
            use_graph_capture=use_graph_capture,
            goal_joint_pos=goal_joint_pos,
        )

        print("gravity", self.model.gravity, self.gravity)
        self.hand_target_ke = self.model.joint_target_ke
        self.hand_target_kd = self.model.joint_target_kd

        self.simulate_params["ag_return_body"] = True

    @property
    def base_joint(self):
        base_joint = ""
        if self.fix_position and self.fix_orientation:
            base_joint = None
        elif self.fix_orientation:
            base_joint += "rx, ry, rz "
        elif self.fix_position:
            base_joint = "px, py, pz"
        else:
            base_joint = ""
            self.floating_base = True
        return base_joint

    def init_sim(self):
        super().init_sim()
        # create mapping from body name to index
        self.body_name_to_idx, self.joint_name_to_idx = {}, {}
        for i, body_name in enumerate(self.model.body_name):
            body_ind = self.body_name_to_idx.get(body_name, [])
            body_ind.append(i)
            self.body_name_to_idx[body_name] = body_ind

        # create mapping from joint name to index
        for i, joint_name in enumerate(self.model.joint_name):
            joint_ind = self.joint_name_to_idx.get(joint_name, [])
            joint_ind.append(i)
            self.joint_name_to_idx[joint_name] = joint_ind

        self.body_name_to_idx = {k: np.array(v) for k, v in self.body_name_to_idx.items()}
        self.joint_name_to_idx = {k: np.array(v) for k, v in self.joint_name_to_idx.items()}

    def _get_obs_dict(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        obs_dict = {}
        obs_dict["hand_joint_pos"] = joint_q[:, : self.hand_num_joint_axis]
        obs_dict["hand_joint_vel"] = joint_qd[:, : self.hand_num_joint_axis]
        if self.object_type is not None:
            obs_dict["object_joint_pos"] = joint_q[
                :, self.object_joint_start : self.object_joint_start + self.object_num_joint_axis
            ]
            obs_dict["object_joint_vel"] = joint_qd[
                :, self.object_joint_start : self.object_joint_start + self.object_num_joint_axis
            ]
            obs_dict["goal_joint_pos"] = self.goal_joint_pos.view(self.num_envs, -1)

        obs_dict["target_qpos"] = self.actions.view(self.num_envs, -1)
        obs_dict["hand_qpos"] = self.joint_q.view(self.num_envs, -1)[:, self.env_joint_target_indices]
        self.extras["obs_dict"] = obs_dict
        return obs_dict

    def sample_grasps(self, num_envs):
        self.grasp = self.grasps[np.random.randint(len(self.grasps), size=num_envs)]
        self.hand_init_xform = np.stack([g.xform for g in self.grasp], axis=0)
        self.hand_init_q = np.stack([g.q for g in self.grasp], axis=0)

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        # need to set the base joint of each env to sampled grasp xform
        # then set each joint target pos to grasp.
        try:
            joint_q[env_ids], joint_qd[env_ids] = super().get_stochastic_init(env_ids, joint_q, joint_qd)
        except:
            pass
        if self.grasps is not None:
            assert joint_q.shape[-1] == self.hand_init_q.shape[-1]
            joint_q[env_ids, self.env_joint_target_indices] = self.hand_init_q.copy()
            self._set_hand_base_xform(env_ids, self.hand_init_xform)
            __import__("ipdb").set_trace()

        return joint_q[env_ids], joint_qd[env_ids]

    def _set_hand_base_xform(self, env_ids, xform):
        joint_X_p = wp.to_torch(self.model.joint_X_p).view(self.num_envs, -1, 7)
        joint_X_p[env_ids, self.hand_joint_start] = xform
        self.model.joint_X_p.assign(wp.from_torch(joint_X_p.view(-1, 7), dtype=wp.transform)),
        # eval_fk should be run after this in reset, do not call this function directly

    def reset(self, env_ids=None, force_reset=True):
        if self.grasps is not None:
            self.sample_grasps()
        return super().reset(env_ids=env_ids, force_reset=force_reset)

    def create_articulation(self, builder):
        if self.hand_type == HandType.ALLEGRO:
            bu.create_allegro_hand(
                builder,
                self.action_type,
                stiffness=self.hand_stiffness,
                damping=self.hand_damping,
                base_joint=self.base_joint,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
                collapse_joints=self.collapse_joints,
            )
        elif self.hand_type == HandType.SHADOW:
            bu.create_shadow_hand(
                builder,
                self.action_type,
                stiffness=self.hand_stiffness,
                damping=self.hand_damping,
                base_joint=self.base_joint,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
                collapse_joints=self.collapse_joints,
            )
        else:
            raise NotImplementedError("Hand type not supported:", self.hand_type)

        self.hand_joint_names = builder.joint_name[:]
        self.hand_num_joint_axis = builder.joint_axis_count
        self.num_joint_q += len(builder.joint_q)
        self.num_joint_qd += len(builder.joint_qd)
        valid_joint_types = supported_joint_types[self.action_type]
        hand_env_joint_mask = list(
            map(lambda x: x[0], filter(lambda x: x[1] in valid_joint_types, enumerate(builder.joint_type)))
        )
        if len(hand_env_joint_mask) > 0:
            joint_indices = []
            for i in hand_env_joint_mask:
                joint_start, axis_count = builder.joint_axis_start[i], joint_coord_map[builder.joint_type[i]]
                joint_indices.append(np.arange(joint_start, joint_start + axis_count))
            joint_indices = np.concatenate(joint_indices)
        else:
            joint_indices = []

        if self.object_type:
            object_articulation_builder = builder  # wp.sim.ModelBuilder()
            super().create_articulation(object_articulation_builder)
            # self.object_num_joint_axis = object_articulation_builder.joint_axis_count - self.hand_num_joint_axis
            # self.object_num_joint_axis = object_articulation_builder.joint_axis_count
            # self.object_joint_start = self.hand_num_joint_axis
            # self.asset_builders.insert(0, object_articulation_builder)

        self.env_joint_mask = hand_env_joint_mask
        self.env_joint_target_indices = joint_indices
        self.hand_joint_start = joint_indices[0]
        assert self.num_acts == len(joint_indices), "num_act must match number of joint control indices"
        self.env_num_joints = len(joint_indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_type", type=str, default="allegro")
    parser.add_argument("--action_type", type=str, default="position")
    parser.add_argument("--object_type", type=str, default=None)
    parser.add_argument("--object_id", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--norender", action="store_false", dest="render")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_obs", type=int, default=36)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--stiffness", type=float, default=5000.0)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--grasp_file", type=str, default="")
    parser.add_argument("--grasp_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_steps", default=100, type=int)
    parser.set_defaults(render=True)

    args = parser.parse_args()
    if args.debug:
        wp.config.mode = "debug"
        wp.config.print_launches = True
        wp.config.verify_cuda = True

    if args.object_type is None:
        object_type = None
    else:
        object_type = ObjectType[args.object_type.upper()]

    if args.headless:
        HandObjectTask.opengl_render_settings["headless"] = True

    rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}
    HandObjectTask.profile = args.profile

    env = HandObjectTask(
        args.num_envs,
        args.num_obs,
        args.episode_length,
        action_type=ActionType[args.action_type.upper()],
        object_type=object_type,
        object_id=args.object_id,
        hand_type=HandType[args.hand_type.upper()],
        render=args.render,
        grasp_file=args.grasp_file,
        grasp_id=args.grasp_id,
        stiffness=args.stiffness,
        damping=args.damping,
        reward_params=rew_params,
    )
    if args.headless and args.render:
        from .wrappers import Monitor

        env = Monitor(env, "outputs/videos")
    run_env(env, num_steps=args.num_steps)
    env.close()
