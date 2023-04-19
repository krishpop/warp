import warp as wp
import warp.sim
import warp.sim.render
import warp.sim.tiny_render
import numpy as np
import math
import argparse
import os
from enum import Enum
from tqdm import trange
from typing import Tuple

wp.init()


class RenderMode(Enum):
    NONE = "none"
    TINY = "tiny"
    USD = "usd"

    def __str__(self):
        return self.value


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"

    def __str__(self):
        return self.value


class Environment:
    env_name: str = "WarpEnv"

    frame_dt = 1.0 / (60.0)

    episode_duration = 15.0  # seconds
    episode_frames = int(episode_duration / frame_dt)

    # whether to play the simulation indefinitely when using the Tiny renderer
    continuous_tiny_render: bool = True

    sim_substeps_euler: int = 16
    sim_substeps_xpbd: int = 5

    euler_settings = dict()
    xpbd_settings = dict()

    render_mode: RenderMode = RenderMode.USD
    tiny_render_settings = dict()
    usd_render_settings = dict(scaling=10.0)

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True

    profile: bool = False

    use_graph_capture: bool = wp.get_preferred_device().is_cuda
    device: str = "cuda"

    num_envs: int = 100

    activate_ground_plane: bool = True

    integrator_type: IntegratorType = IntegratorType.XPBD

    upaxis: str = "y"
    gravity: float = -9.81
    env_offset: Tuple[float, float, float] = (5.0, 0.0, 5.0)

    # stiffness and damping for joint attachment dynamics used by Euler
    joint_attach_ke: float = 10000.0
    joint_attach_kd: float = 10.0

    # distance threshold at which contacts are generated
    rigid_contact_margin: float = 0.05
    # maximal number of contacts per shape mesh
    rigid_mesh_contact_max: int = 1000
    # render rigid contacts
    show_rigid_contact_points: bool = False
    # render models in different collision groups
    separate_collision_group: bool = True

    plot_body_coords: bool = False
    plot_joint_coords: bool = False

    requires_grad: bool = False

    def __init__(self):
        self.stage_path = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--integrator",
            help="Type of integrator",
            type=IntegratorType,
            choices=list(IntegratorType),
            default=self.integrator_type.value,
        )
        self.parser.add_argument(
            "--visualizer",
            help="Type of renderer",
            type=RenderMode,
            choices=list(RenderMode),
            default=self.render_mode.value,
        )
        self.parser.add_argument(
            "--num_envs",
            help="Number of environments to simulate",
            type=int,
            default=self.num_envs,
        )
        self.parser.add_argument(
            "--profile",
            help="Enable profiling",
            default=self.profile,
            action="store_true",
        )

    def parse_args(self):
        args = self.parser.parse_args()
        self.integrator_type = args.integrator
        self.render_mode = args.visualizer
        self.num_envs = args.num_envs
        self.profile = args.profile
        self.stage_path = f"{self.env_name}_{self.num_envs}"

    def init(self):
        if self.integrator_type == IntegratorType.EULER:
            self.sim_substeps = self.sim_substeps_euler
        elif self.integrator_type == IntegratorType.XPBD:
            self.sim_substeps = self.sim_substeps_xpbd

        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = int(self.episode_duration / self.sim_dt)

        self.sim_time = 0.0
        self.render_time = 0.0

        self.builder = wp.sim.ModelBuilder(gravity=self.gravity)
        try:
            articulation_builder = wp.sim.ModelBuilder()
            self.create_articulation(articulation_builder)
            env_offsets = wp.sim.tiny_render.compute_env_offsets(
                self.num_envs, self.env_offset, self.upaxis
            )
            for i in range(self.num_envs):
                xform = wp.transform(env_offsets[i], wp.quat_identity())
                if self.render_mode == RenderMode.USD:
                    quat_rotate = wp.quat(0, 0, 0, 1.0)
                    if self.upaxis == "y":
                        quat_rotate = wp.quat_from_axis_angle(
                            (1.0, 0.0, 0.0), -math.pi * 0.5
                        )
                    xform = wp.transform(env_offsets[i], quat_rotate)
                # self.builder.add_builder(articulation_builder, xform)
                self.builder.add_rigid_articulation(
                    articulation_builder,
                    xform,
                    separate_collision_group=self.separate_collision_group,
                )
            self.bodies_per_env = len(articulation_builder.body_q)
        except NotImplementedError:
            # custom simulation setup where something other than an articulation is used
            self.setup(self.builder)
            self.bodies_per_env = len(self.builder.body_q)

        if not self.activate_ground_plane:
            self.model = self.builder.finalize(device=self.device)
        else:
            self.model = self.builder.finalize(
                device=self.device,
                rigid_mesh_contact_max=self.rigid_mesh_contact_max,
                requires_grad=self.requires_grad,
            )
        #     requires_grad=self.requires_grad,
        # )
        self.model.ground = self.activate_ground_plane
        self.model.joint_q.requires_grad = True
        self.model.joint_qd.requires_grad = True
        self.model.joint_act.requires_grad = True
        self.model.body_q.requires_grad = True
        self.model.body_qd.requires_grad = True

        self.model.joint_attach_ke = self.joint_attach_ke
        self.model.joint_attach_kd = self.joint_attach_kd

        if self.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator(**self.euler_settings)
        elif self.integrator_type == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(**self.xpbd_settings)

        if self.render_mode != RenderMode.NONE:
            self.initialize_renderer()
        else:
            self.renderer = None

    def initialize_renderer(self, stage_path=None):
        if stage_path is not None:
            self.stage_path = stage_path
        if not self.stage_path.endswith(".usd"):
            self.stage_path += ".usd"

        if os.path.dirname(self.stage_path) == "":
            self.stage_path = "outputs/" + self.stage_path

        if not os.path.exists(os.path.dirname(self.stage_path)):
            os.makedirs(os.path.dirname(self.stage_path))

        if self.render_mode == RenderMode.USD:
            print("Initializing renderer writing to path: {}".format(self.stage_path))
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                self.stage_path,
                upaxis=self.upaxis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                **self.usd_render_settings,
            )
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
        elif self.render_mode == RenderMode.TINY:
            # self.writer = imageio.get_writer(f"outputs/{self.stage_path}.mp4", fps=30)
            self.renderer = wp.sim.render.SimRendererTiny(
                self.model,
                self.env_name,  # window name
                upaxis="z",
                # env_offset=self.env_offset,
                **self.tiny_render_settings,
            )

        self.render_time = 0.0
        self.render_freq = 10

    def create_articulation(self, builder):
        raise NotImplementedError

    def setup(self, builder):
        pass

    def customize_model(self, model):
        pass

    def before_simulate(self):
        pass

    def custom_update(self):
        pass

    def update(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.custom_update()
            wp.sim.collide(self.model, self.state_0)
            self.state_1 = self.integrator.simulate(
                self.model, self.state_0, self.state_1, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def run(self):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.render_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if self.eval_fk:
            wp.sim.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0
            )

        self.before_simulate()

        if self.renderer is not None:
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

            if self.render_mode == RenderMode.TINY and self.renderer.start_paused:
                self.renderer.paused = True

        profiler = {}

        if self.use_graph_capture:
            # create update graph
            wp.capture_begin()

            # simulate
            self.update()
            graph = wp.capture_end()
        else:
            if self.plot_body_coords:
                q_history = []
                q_history.append(self.state_0.body_q.numpy().copy())
                qd_history = []
                qd_history.append(self.state_0.body_qd.numpy().copy())
                delta_history = []
                delta_history.append(self.state_0.body_deltas.numpy().copy())
                num_con_history = []
                num_con_history.append(
                    self.model.rigid_contact_inv_weight.numpy().copy()
                )
            if self.plot_joint_coords:
                joint_q_history = []
                joint_q = wp.zeros_like(self.model.joint_q)
                joint_qd = wp.zeros_like(self.model.joint_qd)

        # simulate
        with wp.ScopedTimer(
            "simulate", detailed=False, print=False, active=True, dict=profiler
        ):
            if self.renderer is not None:
                with wp.ScopedTimer("render", False):
                    self.render_time += self.frame_dt

                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(self.state_0)
                    self.renderer.end_frame()

            while True:
                for f in trange(self.episode_frames):
                    if self.use_graph_capture:
                        wp.capture_launch(graph)
                    else:
                        for i in range(0, self.sim_substeps):
                            self.state_0.clear_forces()
                            wp.sim.collide(self.model, self.state_0)
                            self.state_1 = self.integrator.simulate(
                                self.model,
                                self.state_0,
                                self.state_1,
                                self.sim_dt,
                                requires_grad=self.requires_grad,
                            )
                            self.state_0, self.state_1 = self.state_1, self.state_0
                            self.sim_time += self.sim_dt

                    if self.renderer is not None:
                        with wp.ScopedTimer("render", False):
                            self.render_time += self.frame_dt  # * 300.0

                            self.renderer.begin_frame(self.render_time)
                            self.renderer.render(self.state_0)

                            if False and self.max_contact_count > 0:
                                self.renderer.render_points(
                                    "contact_points_a",
                                    np.array(self.points_a),
                                    radius=0.05,
                                )
                                self.renderer.render_points(
                                    "contact_points_b",
                                    np.array(self.points_b),
                                    radius=0.05,
                                )

                            self.renderer.end_frame()

                if (
                    not self.continuous_tiny_render
                    or self.render_mode != RenderMode.TINY
                ):
                    break

            wp.synchronize()

        avg_time = np.array(profiler["simulate"]).mean() / self.episode_frames
        avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

        print(
            f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}"
        )

        if self.renderer is not None:
            self.renderer.save()

        if self.plot_body_coords:
            import matplotlib.pyplot as plt

            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)
            # print("max num_con_history:", np.max(num_con_history))

            body_indices = [9]

            fig, ax = plt.subplots(
                len(body_indices), 7, figsize=(10, 10), squeeze=False
            )
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i, 0].set_title(f"Body {j} Position")
                ax[i, 0].grid()
                ax[i, 1].set_title(f"Body {j} Orientation")
                ax[i, 1].grid()
                ax[i, 2].set_title(f"Body {j} Linear Velocity")
                ax[i, 2].grid()
                ax[i, 3].set_title(f"Body {j} Angular Velocity")
                ax[i, 3].grid()
                ax[i, 4].set_title(f"Body {j} Linear Delta")
                ax[i, 4].grid()
                ax[i, 5].set_title(f"Body {j} Angular Delta")
                ax[i, 5].grid()
                ax[i, 6].set_title(f"Body {j} Num Contacts")
                ax[i, 6].grid()
                ax[i, 0].plot(q_history[:, j, :3])
                ax[i, 1].plot(q_history[:, j, 3:])
                ax[i, 2].plot(qd_history[:, j, 3:])
                ax[i, 3].plot(qd_history[:, j, :3])
                ax[i, 4].plot(delta_history[:, j, 3:])
                ax[i, 5].plot(delta_history[:, j, :3])
                ax[i, 6].plot(num_con_history[:, j])
                ax[i, 0].set_xlim(0, self.sim_steps)
                ax[i, 1].set_xlim(0, self.sim_steps)
                ax[i, 2].set_xlim(0, self.sim_steps)
                ax[i, 3].set_xlim(0, self.sim_steps)
                ax[i, 4].set_xlim(0, self.sim_steps)
                ax[i, 5].set_xlim(0, self.sim_steps)
                ax[i, 6].set_xlim(0, self.sim_steps)
                ax[i, 6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if self.plot_joint_coords:
            import matplotlib.pyplot as plt

            joint_q_history = np.array(joint_q_history)
            dof_q = joint_q_history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof_q)))
            nrows = int(np.ceil(dof_q / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )

            joint_id = 0
            joint_names = {
                wp.sim.JOINT_BALL.val: "ball",
                wp.sim.JOINT_REVOLUTE.val: "hinge",
                wp.sim.JOINT_PRISMATIC.val: "slide",
                wp.sim.JOINT_UNIVERSAL.val: "universal",
                wp.sim.JOINT_COMPOUND.val: "compound",
                wp.sim.JOINT_FREE.val: "free",
                wp.sim.JOINT_FIXED.val: "fixed",
                wp.sim.JOINT_DISTANCE.val: "distance",
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while (
                joint_id < len(joint_type) - 1
                and joint_type[joint_id] == wp.sim.JOINT_FIXED.val
            ):
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            qd_i = qd_start[joint_id]
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof_q:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(joint_q_history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE.val:
                    lower = joint_lower[qd_i]
                    if abs(lower) < 2 * np.pi:
                        ax.axhline(lower, color="red")
                    upper = joint_upper[qd_i]
                    if abs(upper) < 2 * np.pi:
                        ax.axhline(upper, color="red")
                joint_name = joint_names[joint_type[joint_id]]
                ax.set_title(
                    f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})"
                )
                if (
                    joint_id < self.model.joint_count - 1
                    and q_start[joint_id + 1] == dim + 1
                ):
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()

        return 1000.0 * float(self.num_envs) / avg_time


def run_demo(Demo):
    demo = Demo()
    demo.parse_args()
    if demo.profile:
        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            demo = Demo()
            demo.parse_args()
            demo.init()
            steps_per_second = demo.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

        # plot
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()
    else:
        demo.init()
        return demo.run()
