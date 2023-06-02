import warp as wp
import warp.sim
import warp.sim.render

from typing import Tuple
import numpy as np
import os
import math

from env.environment import Environment, RenderMode, IntegratorType, run_env

wp.init()


class PandaSimulation(Environment):
    sim_name: str = "example_sim_panda"

    frame_dt = 1.0 / 60.0

    episode_duration = 4.0      # seconds
    episode_frames = int(episode_duration / frame_dt)

    render_mode: RenderMode = RenderMode.OPENGL

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True

    profile: bool = False

    # XXX important to be able to manually set the controls at each step
    use_graph_capture: bool = False

    activate_ground_plane: bool = True

    integrator_type: IntegratorType = IntegratorType.XPBD

    upaxis: str = "y"
    gravity: float = -9.81
    env_offset: Tuple[float, float, float] = (5.0, 0.0, 5.0)

    show_rigid_contact_points = True
    contact_points_radius = 1e-4

    ########

    env_offset = (6.0, 0.0, 6.0)
    opengl_render_settings = dict(scaling=10.0)
    usd_render_settings = dict(scaling=200.0)

    sim_substeps_euler = 64
    sim_substeps_xpbd = 15

    num_envs = 1

    xpbd_settings = dict(
        iterations=5,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=0.9,
        rigid_contact_con_weighting=True,
        enable_restitution=True,
    )

    rigid_contact_margin = 0.001

    # contact thickness to apply around mesh shapes
    contact_thickness = 1e-3

    frame_counter = 0

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/panda/panda_gripper.urdf"),
            builder,
            # xform=wp.transform(np.array([(i//10)*1.0, 0.30, (i%10)*1.0]), wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5)),
            xform=wp.transform(np.array([0., 0.30, 0.]),
                               wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.5)),
            floating=False,
            base_joint="px, py, pz, ry",
            density=1000,
            armature=0.001,
            stiffness=0.0,  # 120,
            damping=10,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.5,
            shape_thickness=self.contact_thickness,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=True,
            enable_self_collisions=False)

        for mesh in builder.shape_geo_src:
            if isinstance(mesh, wp.sim.Mesh):
                mesh.remesh(visualize=False)

        # use velocity drive for all joints
        for i in range(builder.joint_axis_count):
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_TARGET_VELOCITY
            builder.joint_target_ke[i] = 1e4
        # lower target stiffness for the fingers
        builder.joint_target_ke[4] = 1e3
        builder.joint_target_ke[5] = 1e3

        # Stick dims
        hx = 0.3
        hy = 0.02
        hz = 0.015
        body = builder.add_body(origin=wp.transform((0., hy + self.contact_thickness, 0.), wp.quat_identity()))
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, density=1., ke=1.e5, kf=500.0,
                              kd=250, mu=1.5, thickness=self.contact_thickness)

        self.dof_q_per_env = len(builder.joint_q)

    def update(self):
        # set joint velocity targets
        joint_target = self.model.joint_target.numpy().reshape(self.num_envs, self.dof_q_per_env)
        joint_target[:, :3] = 0.0
        if self.frame_counter < 60:
            # move hand downwards
            joint_target[:, 1] = -0.1
            # open fingers
            joint_target[:, 4] = 0.3  # left finger
            joint_target[:, 5] = 0.3  # right finger
        elif self.frame_counter < 100:
            # close fingers
            joint_target[:, 4] = -1.3  # left finger
            joint_target[:, 5] = -1.3  # right finger
            # stop moving downwards
            joint_target[:, 1] = 0.1
        elif self.frame_counter < 150:
            # keep some pressure on the grasp
            joint_target[:, 4] = -0.1  # left finger
            joint_target[:, 5] = -0.1  # right finger
            # move hand upwards
            joint_target[:, 1] = 0.9
        else:
            # keep some pressure on the grasp
            joint_target[:, 4] = -0.1  # left finger
            joint_target[:, 5] = -0.1  # right finger
            # prevent hand from falling due to insufficient gravity compensation
            joint_target[:, 1] = 0.4

        self.model.joint_target = wp.array(joint_target.flatten(), dtype=wp.float32, device=self.device)

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        self.frame_counter += 1


run_env(PandaSimulation)
