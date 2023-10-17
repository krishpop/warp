# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# MPC toolbox
#
###########################################################################

from warp.sim.integrator_xpbd import apply_joint_torques
from warp.tests.grad_utils import check_tape_safety, check_backward_pass, check_jacobian, function_jacobian_fd
from warp.optim import Adam, SGD
from environment import RenderMode
import sys
import numpy as np
import warp as wp
from enum import Enum

from tqdm import trange

import matplotlib.pyplot as plt

DEBUG_PLOTS = True

# wp.config.verify_cuda = True
# wp.config.mode = "debug"
# wp.config.verify_fp = True

wp.init()
# wp.set_device("cpu")


if DEBUG_PLOTS:

    import pyqtgraph as pg
    from PyQt5 import QtWidgets

    class MainWindow(QtWidgets.QMainWindow):

        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            self.layoutWidget = pg.LayoutWidget()
            self.layoutWidget.addWidget

            # self.graphWidget = pg.PlotWidget()
            self.graphWidget1 = pg.GraphicsLayoutWidget()
            self.graphWidget2 = pg.GraphicsLayoutWidget()
            self.layoutWidget.addWidget(self.graphWidget1)
            self.layoutWidget.addWidget(self.graphWidget2)
            self.setCentralWidget(self.layoutWidget)


class InterpolationMode(Enum):
    INTERPOLATE_HOLD = "hold"
    INTERPOLATE_LINEAR = "linear"
    INTERPOLATE_CUBIC = "cubic"

    def __str__(self):
        return self.value


# Types of action interpolation
INTERPOLATE_HOLD = wp.constant(0)
INTERPOLATE_LINEAR = wp.constant(1)
INTERPOLATE_CUBIC = wp.constant(2)


@wp.kernel
def replicate_states(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    bodies_per_env: int,
    # outputs
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    env_offset = tid * bodies_per_env
    for i in range(bodies_per_env):
        body_q_out[env_offset + i] = body_q_in[i]
        body_qd_out[env_offset + i] = body_qd_in[i]


@wp.kernel
def sample_gaussian(
    mean_trajectory: wp.array(dtype=float, ndim=3),
    noise_scale: float,
    num_control_points: int,
    control_dim: int,
    control_limits: wp.array(dtype=float, ndim=2),
    # outputs
    seed: wp.array(dtype=int),
    rollout_trajectories: wp.array(dtype=float, ndim=3),
):
    env_id, point_id, control_id = wp.tid()
    unique_id = (env_id * num_control_points + point_id) * control_dim + control_id
    r = wp.rand_init(seed[0], unique_id)
    mean = mean_trajectory[0, point_id, control_id]
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    sample = mean + noise_scale * wp.randn(r)
    for i in range(10):
        if sample < lo or sample > hi:
            sample = mean + noise_scale * wp.randn(r)
        else:
            break
    rollout_trajectories[env_id, point_id, control_id] = wp.clamp(sample, lo, hi)
    seed[0] = seed[0] + 1


@wp.kernel
def interpolate_control_hold(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    # outputs
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    control = control_points[env_id, t_id, control_id]
    torque_id = env_id * torque_dim + control_dims[control_id]
    torques[torque_id] = control * control_gains[control_id]


@wp.kernel
def interpolate_control_linear(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    # outputs
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    frac = t - wp.floor(t)
    control_left = control_points[env_id, t_id, control_id]
    control_right = control_points[env_id, t_id + 1, control_id]
    torque_id = env_id * torque_dim + control_dims[control_id]
    action = control_left * (1.0 - frac) + control_right * frac
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def interpolate_control_cubic(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    dt: float,
    torque_dim: int,
    # outputs
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    q = t - wp.floor(t)
    c0 = control_points[env_id, t_id, control_id]
    c1 = control_points[env_id, t_id + 1, control_id]
    c2 = control_points[env_id, t_id + 2, control_id]
    c3 = control_points[env_id, t_id + 3, control_id]
    # Eq. 17
    phi_0 = 0.5 * ((c2 - c1) + (c1 - c0)) / dt
    # Eq. 18
    phi_1 = 0.5 * ((c3 - c2) + (c2 - c1)) / dt
    # Eq. 20-23
    a = 2.0 * q ** 3.0 - 3.0 * q ** 2.0 + 1.0
    b = (q ** 3.0 - 2.0 * q ** 2.0 + q) * dt
    c = -2.0 * q ** 3.0 + 3.0 * q ** 2.0
    d = (q ** 3.0 - q ** 2.0) * dt
    torque_id = env_id * torque_dim + control_dims[control_id]
    action = a * c1 + b * phi_0 + c * c2 + d * phi_1
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def control_to_body_force(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    bodies_per_env: int,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    control_left = control_points[env_id, t_id, control_id]
    force_id = env_id * bodies_per_env + control_dims[control_id]
    c = control_left * control_gains[control_id]
    lin_f = wp.vec3(c, 0.0, 0.0)
    ang_f = wp.vec3(0.0, 0.0, 0.0)
    body_f[force_id] = wp.spatial_vector(ang_f, lin_f)


@wp.kernel
def pick_best_trajectory(
    rollout_trajectories: wp.array(dtype=float, ndim=3),
    lowest_cost_id: int,
    # outputs
    best_traj: wp.array(dtype=float, ndim=3),
):
    t_id, control_id = wp.tid()
    best_traj[0, t_id, control_id] = rollout_trajectories[lowest_cost_id, t_id, control_id]


@wp.kernel
def enforce_control_limits(
    control_points: wp.array(dtype=float, ndim=3),
    control_limits: wp.array(dtype=float, ndim=2),
):
    env_id, t_id, control_id = wp.tid()
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    control_points[env_id, t_id, control_id] = wp.clamp(
        control_points[env_id, t_id, control_id], lo, hi
    )


class Controller:

    noise_scale = 0.1

    interpolation_mode = InterpolationMode.INTERPOLATE_LINEAR
    # interpolation_mode = InterpolationMode.INTERPOLATE_CUBIC
    # interpolation_mode = InterpolationMode.INTERPOLATE_HOLD

    def __init__(self, env_fn):

        # total number of time steps in the trajectory that is optimized
        self.traj_length = 15000

        # time steps between control points
        self.control_step = 30
        # number of control horizon points to interpolate between
        self.num_control_points = 3
        # total number of horizon time steps
        self.horizon_length = self.num_control_points * self.control_step
        # number of trajectories to sample for optimization
        self.num_threads = 100
        # number of steps to follow before optimizing again
        self.optimization_interval = 1

        # whether env_rollout requires gradients
        self.use_diff_sim = False

        # create environment for sampling trajectories for optimization
        self.env_rollout = env_fn()
        self.env_rollout.num_envs = self.num_threads
        self.env_rollout.render_mode = RenderMode.NONE
        # self.env_rollout.render_mode = RenderMode.OPENGL
        self.env_rollout.requires_grad = self.use_diff_sim
        self.env_rollout.episode_frames = self.horizon_length
        self.env_rollout.init()

        # create environment for visualization and the reference state
        self.env_ref = env_fn()
        self.env_ref.num_envs = 1
        # self.env_ref.requires_grad = self.use_diff_sim
        self.env_ref.episode_frames = self.traj_length
        # self.env_ref.render_mode = RenderMode.NONE
        self.env_ref.init()
        self.dof_count = len(self.env_ref.control)

        self.use_graph_capture = wp.get_device(self.device).is_cuda

        # optimized control points for the current horizon
        self.best_traj = None
        # control point samples for the current horizon
        self.rollout_trajectories = None

        assert len(self.env_rollout.controllable_dofs) == len(self.env_rollout.control_gains)
        assert len(self.env_rollout.controllable_dofs) == len(self.env_rollout.control_limits)

        # construct Warp array for the indices of controllable dofs
        self.controllable_dofs = wp.array(self.env_rollout.controllable_dofs, dtype=int)
        self.control_gains = wp.array(self.env_rollout.control_gains, dtype=float)
        self.control_limits = wp.array(self.env_rollout.control_limits, dtype=float)

        self.sampling_seed_counter = wp.zeros(1, dtype=int)

        self.controllable_dofs_np = np.array(self.env_rollout.controllable_dofs)

        # CUDA graphs
        self._opt_graph = None

        self._optimizer = None

        self.plotting_app = None
        self.plotting_window = None
        if DEBUG_PLOTS:
            self.plotting_app = QtWidgets.QApplication(sys.argv)
            self.plotting_window = MainWindow()
            self.plotting_window.show()

            # maximum number of threads to visualize in the control sample views
            self.max_plotting_threads = 50

            # get matplotlib colors as list for tab20 colormap
            self.plot_colors = (plt.get_cmap("tab10")(np.arange(10, dtype=int))[:, :3] * 255).astype(int)

            self.data_xs = np.arange(self.num_control_points)
            ys = np.zeros(self.num_control_points)
            self.rollout_plots = []
            self.rollout_plot_axs = []
            num_plots = self.control_dim + 1
            ncols = int(np.ceil(np.sqrt(num_plots)))
            nrows = int(np.ceil(num_plots / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )
            for i in range(num_plots):
                p = self.plotting_window.graphWidget1.addPlot(row=i // ncols, col=i % ncols)
                if i == 0:
                    p.setTitle("Cost")
                    # p.setYRange(0.0, 10.0)
                    self.cost_plot = p.plot(np.arange(self.num_threads), np.zeros(
                        self.num_threads), symbol='o', symbolSize=4)
                else:
                    p.setTitle(f"Control {i-1}")
                    p.setYRange(*self.env_rollout.control_limits[i - 1])
                self.rollout_plot_axs.append(p)
            plotting_threads = min(self.num_threads, self.max_plotting_threads)
            for i in range(plotting_threads):
                thread_plots = []
                for j in range(1, self.control_dim + 1):
                    pen = pg.mkPen(color=self.plot_colors[i % len(self.plot_colors)])
                    p = self.rollout_plot_axs[j].plot(self.data_xs, ys, pen=pen, symbol='o', symbolSize=2)
                    thread_plots.append(p)
                self.rollout_plots.append(thread_plots)

            # self.plotting_window2 = MainWindow()
            # self.plotting_window2.show()

            # get matplotlib colors as list for tab20 colormap
            self.plot_colors = (plt.get_cmap("tab10")(np.arange(10, dtype=int))[:, :3] * 255).astype(int)

            self.data_xs = np.arange(self.num_control_points)
            ys = np.zeros(self.num_control_points)
            self.ref_plots = []
            self.ref_plot_axs = []
            ncols = int(np.ceil(np.sqrt(num_plots)))
            nrows = int(np.ceil(num_plots / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )
            scaled_control_limits = np.array(self.env_rollout.control_limits) * \
                np.array(self.env_rollout.control_gains)[:, np.newaxis]
            for i in range(num_plots):
                p = self.plotting_window.graphWidget2.addPlot(row=i // ncols, col=i % ncols)
                if i == 0:
                    p.setTitle("Ref Cost")
                    # p.setYRange(0.0, 10.0)
                    self.ref_cost_plot = p.plot(np.arange(self.num_threads), np.zeros(
                        self.num_threads), pen=pg.mkPen(color=(255, 0, 0)))
                else:
                    p.setTitle(f"Ref Control {i-1}")
                    p.setYRange(*scaled_control_limits[i - 1])
                self.ref_plot_axs.append(p)
            for j in range(self.control_dim):
                pen = pg.mkPen(color=self.plot_colors[0])
                p = self.ref_plot_axs[j + 1].plot(self.data_xs, ys, pen=pen)
                self.ref_plots.append(p)

    @property
    def control_dim(self):
        return len(self.env_rollout.controllable_dofs)

    @property
    def body_count(self):
        return self.env_ref.bodies_per_env

    @property
    def device(self):
        return self.env_rollout.device

    def allocate_trajectories(self):
        # number of control points that need to be optimized, respective of the type of interpolation
        self.num_control_points_data = self.num_control_points
        if self.interpolation_mode == InterpolationMode.INTERPOLATE_LINEAR:
            self.num_control_points_data += 1
        elif self.interpolation_mode == InterpolationMode.INTERPOLATE_CUBIC:
            self.num_control_points_data += 3
        # optimized control points for the current horizon (3-dimensional to be compatible with rollout trajectories)
        self.best_traj = wp.zeros((1, self.num_control_points_data, self.control_dim),
                                  dtype=float, requires_grad=self.use_diff_sim, device=self.device)
        # control point samples for the current horizon
        self.rollout_trajectories = wp.zeros(
            (self.num_threads, self.num_control_points_data, self.control_dim), dtype=float, requires_grad=self.use_diff_sim, device=self.device)
        # self.rollout_trajectories = wp.array(
        #     [
        #         -np.ones((self.num_control_points_data, self.control_dim)),
        #         np.ones((self.num_control_points_data, self.control_dim)),
        #         np.zeros((self.num_control_points_data, self.control_dim)),
        #     ], dtype=float, requires_grad=self.use_diff_sim, device=self.device)
        # cost of each trajectory
        self.rollout_costs = wp.zeros((self.num_threads,), dtype=float,
                                      requires_grad=self.use_diff_sim, device=self.device)

    def run(self):
        self.env_rollout.before_simulate()
        self.env_ref.reset()
        self.env_ref.before_simulate()
        self.env_ref_acts = []
        self.env_ref_costs = []
        ref_cost = wp.zeros(1, dtype=float, device=self.device)
        assert len(self.env_ref.state.body_q) == self.body_count
        self.allocate_trajectories()
        self.assign_control_fn(self.env_ref, self.best_traj)

        progress = trange(self.traj_length)
        for t in progress:
            # optimize trajectory horizon
            self.optimize(self.env_ref.state)
            # set sim time to zero to execute first optimized action
            self.env_ref.sim_time = 0
            self.env_ref.sim_step = 0
            for _ in range(self.optimization_interval):
                # advance the reference state with the next best action
                self.env_ref.update()
                self.env_ref.render()

            ref_cost.zero_()
            self.env_ref.evaluate_cost(self.env_ref.state, ref_cost, t, self.traj_length)
            self.env_ref_costs.append(ref_cost.numpy()[0])

            if DEBUG_PLOTS:
                self.ref_cost_plot.setData(np.arange(t + 1), self.env_ref_costs)
            #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
            #     fig.suptitle("best traj")
            #     best_traj = self.best_traj.numpy()
            #     for dim in range(self.control_dim):
            #         ax = axes[dim // ncols, dim % ncols]
            #         ax.plot(best_traj[0, :, dim])
            #     plt.show()

            progress.set_description(f"cost: {self.last_lowest_cost:.2f} ({self.last_lowest_cost_id})")

        self.env_ref.after_simulate()
        self.env_rollout.after_simulate()

    def optimize(self, state):
        # predictive sampling algorithm
        if self.use_graph_capture:
            if self._opt_graph is None:
                wp.capture_begin()
                self.sample_controls(self.best_traj)
                self.rollout(state, self.rollout_trajectories)
                self._opt_graph = wp.capture_end()
            else:
                wp.capture_launch(self._opt_graph)
        else:
            self.sample_controls(self.best_traj)
            self.rollout(state, self.rollout_trajectories)
        self.pick_best_control()

    def optimize2(self, state):
        num_opt_steps = 15
        # gradient-based optimization
        if self._optimizer is None:
            # TODO try Adam
            self._optimizer = SGD([self.rollout_trajectories], lr=1e-2, nesterov=False, momentum=0.0)
            # self._optimizer = Adam([self.rollout_trajectories], lr=1e-4)
        if self.use_graph_capture:
            self.sample_controls(self.best_traj)
            if self._opt_graph is None or self.env_ref.invalidate_cuda_graph:
                self.env_rollout.flight_target = self.env_ref.flight_target
                wp.capture_begin()
                self.tape = wp.Tape()
                with self.tape:
                    self.rollout(state, self.rollout_trajectories)
                self.rollout_costs.grad.fill_(1.0)
                self.tape.backward()
                self._optimizer.step([self.rollout_trajectories.grad])
                self.clamp_controls()
                self.tape.zero()
                self._opt_graph = wp.capture_end()
                self.env_ref.invalidate_cuda_graph = False

            for _ in range(num_opt_steps):
                wp.capture_launch(self._opt_graph)
        else:
            self.sample_controls(self.best_traj)
            for it in range(num_opt_steps):
                # check_tape_safety(
                #     lambda s, r: self.rollout(s, r),
                #     inputs=[state, self.rollout_trajectories],
                #     outputs=[self.rollout_costs])
                # check_jacobian(
                #     lambda controls: self.rollout(state, controls),
                #     inputs=[self.rollout_trajectories],
                #     input_names=["controls"],
                #     output_names=["costs"],
                #     plot_jac_on_fail=True
                # )

                use_fd_grads = False

                if not use_fd_grads:
                    self.tape = wp.Tape()
                    with self.tape:
                        self.rollout(state, self.rollout_trajectories)
                        

                    # check_backward_pass(
                    #     self.tape,
                    #     visualize_graph=False,
                    #     analyze_graph=True,
                    #     plot_jac_on_fail=True,
                    #     track_inputs=[self.rollout_trajectories],
                    #     track_outputs=[self.rollout_costs],
                    #     track_input_names=["controls"],
                    #     track_output_names=["costs"],
                    #     # whitelist_kernels={
                    #     #     "apply_joint_torques",
                    #     #     # "solve_simple_body_joints"
                    #     # }
                    # )

                    self.rollout_costs.grad.fill_(1.0)
                    self.tape.backward()
                else:
                    jac_fd = function_jacobian_fd(
                        lambda controls: self.rollout(state, controls),
                        inputs=[self.rollout_trajectories],
                        eps=1e-3)
                    self.rollout_trajectories.grad.assign(jac_fd)

                self._optimizer.step([self.rollout_trajectories.grad])
                lr = 1.5e-3
                print(f"\niter {it} cost:", self.rollout_costs.numpy().flatten())
                # print("\tchecked cost 1:", self.rollout(state, self.rollout_trajectories).numpy())
                # print("\tchecked cost 2:", self.rollout(state, self.rollout_trajectories).numpy())

                # jac_fd = function_jacobian_fd(
                #     lambda controls: self.rollout(state, controls),
                #     inputs=[self.rollout_trajectories],
                #     eps=1e-3)
                # print("\tjac_fd 1:", jac_fd.flatten())
                # jac_fd = function_jacobian_fd(
                #     lambda controls: self.rollout(state, controls),
                #     inputs=[self.rollout_trajectories],
                #     eps=1e-3)
                # print("\tjac_fd 2:", jac_fd.flatten())
                self.rollout_trajectories.assign(self.rollout_trajectories.numpy() -
                                                 lr * self.rollout_trajectories.grad.numpy())
                print("\tgrad:", self.rollout_trajectories.grad.numpy().flatten())
                print("\tcontrols:", self.rollout_trajectories.numpy().flatten())
                self.clamp_controls()
                print("\tclamped controls:", self.rollout_trajectories.numpy().flatten())
                if not use_fd_grads:
                    self.tape.zero()
        wp.synchronize()
        self.pick_best_control()

    def pick_best_control(self):
        costs = self.rollout_costs.numpy()
        lowest_cost_id = np.argmin(costs)
        # print(f"lowest cost: {lowest_cost_id}\t{costs[lowest_cost_id]}")
        self.last_lowest_cost_id = lowest_cost_id
        self.last_lowest_cost = costs[lowest_cost_id]
        wp.launch(
            pick_best_trajectory,
            dim=(self.num_control_points_data, self.control_dim),
            inputs=[self.rollout_trajectories, lowest_cost_id],
            outputs=[self.best_traj],
            device=self.device
        )
        self.rollout_trajectories[-1].assign(self.best_traj[0])

        if DEBUG_PLOTS:
            self.cost_plot.setData(np.arange(self.num_threads), costs)
            trajs = self.rollout_trajectories.numpy()
            plotting_threads = min(self.num_threads, self.max_plotting_threads)
            if len(self.env_ref_acts) > 0:
                env_ref_acts = np.array(self.env_ref_acts)
                for j in range(self.control_dim):
                    for i in range(plotting_threads):
                        self.rollout_plots[i][j].setData(self.data_xs, trajs[i, :self.num_control_points, j])
                    self.ref_plots[j].setData(np.arange(len(env_ref_acts)), env_ref_acts[:, j])

    def clamp_controls(self):
        # enforce control limits on the control points
        wp.launch(
            enforce_control_limits,
            dim=[self.num_threads, self.num_control_points_data, self.control_dim],
            inputs=[self.rollout_trajectories, self.control_limits],
            device=self.device,
        )

    def assign_control_fn(self, env, controls):
        # assign environment control application function that interpolates the control points
        def update_control_hold():
            wp.launch(
                interpolate_control_hold,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.dof_count,
                ],
                outputs=[env.control],
                device=self.device)
            if env == self.env_ref:
                self.env_ref_acts.append(env.control.numpy()[self.controllable_dofs_np])
            if DEBUG_PLOTS and not self.use_graph_capture:
                self.control_hist.append(env.control.numpy()[
                    self.controllable_dofs_np].reshape((-1, self.control_dim)))

        def update_control_linear():
            wp.launch(
                interpolate_control_linear,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.dof_count,
                ],
                outputs=[env.control],
                device=self.device)
            if env == self.env_ref:
                self.env_ref_acts.append(env.control.numpy()[self.controllable_dofs_np])
            if DEBUG_PLOTS and not self.use_graph_capture:
                self.control_hist.append(env.control.numpy()[
                    self.controllable_dofs_np].reshape((-1, self.control_dim)))

        def update_control_cubic():
            wp.launch(
                interpolate_control_cubic,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.control_step * env.frame_dt,
                    self.dof_count,
                ],
                outputs=[env.control],
                device=self.device)
            if env == self.env_ref:
                self.env_ref_acts.append(env.control.numpy()[self.controllable_dofs_np])
            if DEBUG_PLOTS and not self.use_graph_capture:
                self.control_hist.append(env.control.numpy()[
                    self.controllable_dofs_np].reshape((-1, self.control_dim)))

        def update_control_direct():
            wp.launch(
                control_to_body_force,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.dof_count,
                    env.bodies_per_env,
                ],
                outputs=[env.state.body_f],
                device=self.device)

        # env.custom_update = update_control_direct
        # return

        if self.interpolation_mode == InterpolationMode.INTERPOLATE_HOLD:
            env.custom_update = update_control_hold
        elif self.interpolation_mode == InterpolationMode.INTERPOLATE_LINEAR:
            env.custom_update = update_control_linear
        elif self.interpolation_mode == InterpolationMode.INTERPOLATE_CUBIC:
            env.custom_update = update_control_cubic
        else:
            raise NotImplementedError(f"Interpolation mode {self.interpolation_mode} not implemented")

    def rollout(self, state, controls):
        self.env_rollout.reset()
        self.rollout_costs.zero_()
        if DEBUG_PLOTS and not self.use_graph_capture:
            self.control_hist = []

        wp.launch(
            replicate_states,
            dim=self.num_threads,
            inputs=[
                state.body_q,
                state.body_qd,
                self.body_count
            ],
            outputs=[
                self.env_rollout.state.body_q,
                self.env_rollout.state.body_qd
            ],
            device=self.device
        )
        self.assign_control_fn(self.env_rollout, controls)

        for t in range(self.horizon_length):
            self.env_rollout.update()
            self.env_rollout.evaluate_cost(self.env_rollout.state, self.rollout_costs, t, self.horizon_length)
            if not self.use_graph_capture:
                self.env_rollout.render()

        # if DEBUG_PLOTS and not self.use_graph_capture:
        #     self.control_hist = np.array(self.control_hist)
        #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
        #     fig.suptitle("joint acts")
        #     for dim in range(self.control_dim):
        #         ax = axes[dim // ncols, dim % ncols]
        #         ax.plot(self.control_hist[:, :, dim], alpha=0.4)
        #     plt.show()
        #     self.control_hist = []

        return self.rollout_costs

    def sample_controls(self, nominal_traj, noise_scale=noise_scale):
        # sample control waypoints around the nominal trajectory
        # the last trajectory is fixed to the previous best trajectory
        wp.launch(
            sample_gaussian,
            dim=(self.num_threads - 1, self.num_control_points_data, self.control_dim),
            inputs=[
                nominal_traj,
                noise_scale,
                self.num_control_points_data,
                self.control_dim,
                self.control_limits,
                self.sampling_seed_counter,
            ],
            outputs=[self.rollout_trajectories],
            device=self.device
        )
        # if DEBUG_PLOTS:
        #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
        #     fig.suptitle("rollout trajectories")
        #     for dim in range(self.control_dim):
        #         ax = axes[dim // ncols, dim % ncols]
        #         ax.plot(self.rollout_trajectories[:, :, dim].numpy().T, alpha=0.2, label="sampled")
        #         ax.plot(nominal_traj[:, dim].numpy(), label="nominal")
        #     plt.show()

        # self.sampling_seed_counter += self.num_threads * self.num_control_points * self.control_dim

    @staticmethod
    def _create_plot_grid(dof):
        ncols = int(np.ceil(np.sqrt(dof)))
        nrows = int(np.ceil(dof / float(ncols)))
        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            constrained_layout=True,
            figsize=(ncols * 3.5, nrows * 3.5),
            squeeze=False,
            sharex=True,
        )
        for dim in range(ncols * nrows):
            ax = axes[dim // ncols, dim % ncols]
            if dim >= dof:
                ax.axis("off")
                continue
            ax.grid()
        return fig, axes, ncols, nrows


if __name__ == "__main__":
    from env_ant import AntEnvironment
    from env_hopper import HopperEnvironment
    from env_cartpole import CartpoleEnvironment
    from env_drone import DroneEnvironment

    CartpoleEnvironment.env_offset = (0.0, 0.0, 0.0)
    CartpoleEnvironment.single_cartpole = True

    AntEnvironment.env_offset = (0.0, 0.0, 0.0)
    HopperEnvironment.env_offset = (0.0, 0.0, 0.0)
    DroneEnvironment.env_offset = (0.0, 0.0, 0.0)

    # mpc = Controller(AntEnvironment)
    # mpc = Controller(HopperEnvironment)
    mpc = Controller(CartpoleEnvironment)
    # mpc = Controller(DroneEnvironment)

    mpc.run()
