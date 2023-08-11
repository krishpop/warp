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

from random import randint
import os
import sys
import numpy as np
import warp as wp
from environment import RenderMode
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
    from pyqtgraph import PlotWidget, plot
    from PyQt5 import QtWidgets, QtCore


    class MainWindow(QtWidgets.QMainWindow):

        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            # self.graphWidget = pg.PlotWidget()
            self.graphWidget = pg.GraphicsLayoutWidget()
            self.setCentralWidget(self.graphWidget)


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
    control_left = control_points[env_id, t_id, control_id]
    torque_id = env_id * torque_dim + control_dims[control_id]
    torques[torque_id] = control_left * control_gains[control_id]


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
def pick_best_trajectory(
    rollout_trajectories: wp.array(dtype=float, ndim=3),
    lowest_cost_id: int,
    # outputs
    best_traj: wp.array(dtype=float, ndim=3),
):
    t_id, control_id = wp.tid()
    best_traj[0, t_id, control_id] = rollout_trajectories[lowest_cost_id, t_id, control_id]


class Controller:

    noise_scale = 1.1

    interpolation_mode = InterpolationMode.INTERPOLATE_LINEAR
    # interpolation_mode = InterpolationMode.INTERPOLATE_HOLD

    def __init__(self, env_fn):

        # total number of time steps in the trajectory that is optimized
        self.traj_length = 500

        # time steps between control points
        self.control_step = 40
        # number of control horizon points to interpolate between
        self.num_control_points = 3
        # total number of horizon time steps
        self.horizon_length = self.num_control_points * self.control_step
        # number of trajectories to sample for optimization
        self.num_threads = 100

        # create environment for sampling trajectories for optimization
        self.env_rollout = env_fn()
        self.env_rollout.num_envs = self.num_threads
        self.env_rollout.render_mode = RenderMode.NONE
        self.env_rollout.init()

        # create environment for visualization and the reference state
        self.env_ref = env_fn()
        self.env_ref.num_envs = 1
        # self.env_ref.render_mode = RenderMode.NONE
        self.env_ref.init()
        self.dof_count = len(self.env_ref.model.joint_act)

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
        self._rollout_graph = None

        self.use_graph_capture = wp.get_device(self.device).is_cuda  # and not DEBUG_PLOTS

        self.plotting_app = None
        self.plotting_window = None
        if DEBUG_PLOTS:
            self.plotting_app = QtWidgets.QApplication(sys.argv)
            self.plotting_window = MainWindow()
            self.plotting_window.show()

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
                p = self.plotting_window.graphWidget.addPlot(row=i // ncols, col=i % ncols)
                if i == 0:
                    p.setTitle("Cost")
                    # p.setYRange(0.0, 10.0)
                    self.cost_plot = p.plot(np.arange(self.num_threads), np.zeros(self.num_threads))
                else:
                    p.setTitle(f"Control {i-1}")
                    p.setYRange(*self.env_rollout.control_limits[i - 1])
                self.rollout_plot_axs.append(p)
            for i in range(self.num_threads):
                thread_plots = []
                for j in range(1, self.control_dim + 1):
                    pen = pg.mkPen(color=self.plot_colors[i % len(self.plot_colors)])
                    p = self.rollout_plot_axs[j].plot(self.data_xs, ys, pen=pen)
                    thread_plots.append(p)
                self.rollout_plots.append(thread_plots)



                
            self.plotting_window2 = MainWindow()
            self.plotting_window2.show()

            # get matplotlib colors as list for tab20 colormap
            self.plot_colors = (plt.get_cmap("tab10")(np.arange(10, dtype=int))[:, :3] * 255).astype(int)

            self.data_xs = np.arange(self.num_control_points)
            ys = np.zeros(self.num_control_points)
            self.ref_plots = []
            self.ref_plot_axs = []
            num_plots = self.control_dim
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
            scaled_control_limits = np.array(self.env_rollout.control_limits) * np.array(self.env_rollout.control_gains)[:, np.newaxis]
            for i in range(num_plots):
                p = self.plotting_window2.graphWidget.addPlot(row=i // ncols, col=i % ncols)
                p.setTitle(f"Ref Control {i}")
                p.setYRange(*scaled_control_limits[i])
                self.ref_plot_axs.append(p)
            for j in range(self.control_dim):
                pen = pg.mkPen(color=self.plot_colors[0])
                p = self.ref_plot_axs[j].plot(self.data_xs, ys, pen=pen)
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
        # optimized control points for the current horizon (3-dimensional to be compatible with rollout trajectories)
        self.best_traj = wp.zeros((1, self.num_control_points, self.control_dim), dtype=float, device=self.device)
        # control point samples for the current horizon
        self.rollout_trajectories = wp.zeros(
            (self.num_threads, self.num_control_points, self.control_dim), dtype=float, device=self.device)
        # self.rollout_trajectories = wp.array(
        #     [
        #         -np.ones((self.num_control_points, self.control_dim)),
        #         np.ones((self.num_control_points, self.control_dim)),
        #         np.zeros((self.num_control_points, self.control_dim)),
        #     ], dtype=float, device=self.device)
        # cost of each trajectory
        self.rollout_costs = wp.zeros((self.num_threads,), dtype=float, device=self.device)

    def run(self):
        self.env_ref.reset()
        self.env_ref_acts = []
        assert len(self.env_ref.state.body_q) == self.body_count
        self.allocate_trajectories()
        self.assign_control_fn(self.env_ref, self.best_traj)

        progress = trange(self.traj_length)
        for t in progress:
            # optimize trajectory horizon
            self.optimize(self.env_ref.state)
            # self.assign_control_fn(self.env_ref, self.best_traj)
            # set sim time to zero to execute first optimized action
            self.env_ref.sim_time = 0
            # advance the reference state with the next best action
            self.env_ref.update()
            self.env_ref.render()

            # if DEBUG_PLOTS:
            #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
            #     fig.suptitle("best traj")
            #     best_traj = self.best_traj.numpy()
            #     for dim in range(self.control_dim):
            #         ax = axes[dim // ncols, dim % ncols]
            #         ax.plot(best_traj[0, :, dim])
            #     plt.show()

            progress.set_description(f"cost: {self.last_lowest_cost:.2f} ({self.last_lowest_cost_id})")

    def optimize(self, state):
        # predictive sampling algorithm
        if self.use_graph_capture:
            if self._rollout_graph is None:
                wp.capture_begin()
                self.sample_controls(self.best_traj)
                self.rollout(state, self.horizon_length, self.num_threads)
                self._rollout_graph = wp.capture_end()
            else:
                wp.capture_launch(self._rollout_graph)
        else:
            self.sample_controls(self.best_traj)
            self.rollout(state, self.horizon_length, self.num_threads)
        wp.synchronize()
        self.pick_best_control()

    def pick_best_control(self):
        costs = self.rollout_costs.numpy()
        self.cost_plot.setData(np.arange(self.num_threads), costs)
        lowest_cost_id = np.argmin(costs)
        # print(f"lowest cost: {lowest_cost_id}\t{costs[lowest_cost_id]}")
        self.last_lowest_cost_id = lowest_cost_id
        self.last_lowest_cost = costs[lowest_cost_id]
        wp.launch(
            pick_best_trajectory,
            dim=(self.num_control_points, self.control_dim),
            inputs=[self.rollout_trajectories, lowest_cost_id],
            outputs=[self.best_traj],
            device=self.device
        )
        self.rollout_trajectories[-1].assign(self.best_traj[0])

        if DEBUG_PLOTS:
            trajs = self.rollout_trajectories.numpy()
            if len(self.env_ref_acts) > 0:
                env_ref_acts = np.array(self.env_ref_acts)
                for j in range(self.control_dim):
                    for i in range(self.num_threads):
                        self.rollout_plots[i][j].setData(self.data_xs, trajs[i, :, j])
                    self.ref_plots[j].setData(np.arange(len(env_ref_acts)), env_ref_acts[:, j])

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
                outputs=[env.model.joint_act],
                device=self.device)

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
                outputs=[env.model.joint_act],
                device=self.device)
            if env == self.env_ref:
                self.env_ref_acts.append(env.model.joint_act.numpy()[self.controllable_dofs_np])
            if DEBUG_PLOTS and not self.use_graph_capture:
                self.joint_acts.append(env.model.joint_act.numpy()[self.controllable_dofs_np].reshape((-1, self.control_dim)))

        if self.interpolation_mode == InterpolationMode.INTERPOLATE_HOLD:
            env.custom_update = update_control_hold
        elif self.interpolation_mode == InterpolationMode.INTERPOLATE_LINEAR:
            env.custom_update = update_control_linear
        else:
            raise NotImplementedError(f"Interpolation mode {self.interpolation_mode} not implemented")

    def rollout(self, state, num_steps, num_threads):
        self.env_rollout.reset()
        self.rollout_costs.zero_()
        if DEBUG_PLOTS and not self.use_graph_capture:
            self.joint_acts = []

        wp.launch(
            replicate_states,
            dim=(num_threads),
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
        self.assign_control_fn(self.env_rollout, self.rollout_trajectories)

        for t in range(num_steps):
            self.env_rollout.update()
            self.env_rollout.evaluate_cost(self.env_rollout.state, self.rollout_costs)
            if not self.use_graph_capture:
                self.env_rollout.render()

        if DEBUG_PLOTS and not self.use_graph_capture:
            self.joint_acts = np.array(self.joint_acts)
            fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
            fig.suptitle("joint acts")
            for dim in range(self.control_dim):
                ax = axes[dim // ncols, dim % ncols]
                ax.plot(self.joint_acts[:, :, dim], alpha=0.4)
            plt.show()
            self.joint_acts = []

    def sample_controls(self, nominal_traj, noise_scale=noise_scale):
        # sample control waypoints around the nominal trajectory
        # the last trajectory is fixed to the previous best trajectory
        wp.launch(
            sample_gaussian,
            dim=(self.num_threads-1, self.num_control_points, self.control_dim),
            inputs=[
                nominal_traj,
                noise_scale,
                self.num_control_points,
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
    from env_cartpole import CartpoleEnvironment

    CartpoleEnvironment.env_offset = (0.0, 0.0, 0.0)
    CartpoleEnvironment.single_cartpole = True

    mpc = Controller(CartpoleEnvironment)
    mpc.run()

    
    # from env_ant import AntEnvironment

    # AntEnvironment.env_offset = (0.0, 0.0, 0.0)

    # mpc = Controller(AntEnvironment)
    # mpc.run()
