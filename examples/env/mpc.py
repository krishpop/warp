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

import warp as wp
from env.environment import RenderMode

# Types of action interpolation
INTERPOLATE_ZERO = wp.constant(0)
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
        body_q_out[env_offset+i] = body_q_in[i]
        body_qd_out[env_offset+i] = body_qd_in[i]

@wp.kernel
def sample_gaussian(
    mean: wp.array(dtype=wp.float32),
    std: wp.array(dtype=wp.float32),
    # outputs
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    r = wp.rand_init(5483, wp.tid())
    out[tid] = mean[tid] + std[tid] * wp.randn(r)


class Controller:

    noise_scale = 1.0

    def __init__(self, env_fn):


        self.traj_length = 1000

        self.control_step = 10
        self.horizon_length = 50 * self.control_step

        self.num_threads = 500

        self._rollout_trajectories = None

        self.env_rollout = env_fn()
        self.env_rollout.num_envs = self.num_threads
        self.env_rollout.render_mode = RenderMode.NONE
        self.env_rollout.init()

        # create environment for visualization and the reference state
        self.env_ref = env_fn()
        self.env_ref.num_envs = 1
        self.env_ref.init()



    def run(self):
        self.env_rollout.reset()
        state = self.env_ref.reset()
        self.allocate_trajectories()

        for t in range(self.traj_length):

            best_traj = self.optimize(state)

            state = self.advance(best_traj, 1)

    def optimize(self, state):
        # predictive sampling algorithm
        self.rollout(state, self.horizon_length, self.num_threads)
        best_traj = self.select_best(self._rollout_trajectories)
        return best_traj
    
    def rollout(self, state, nominal_traj, num_steps, num_threads):
        controls = self.sample_controls(nominal_traj, num_steps, num_threads)
        for t in range(num_steps):
            pass

    def allocate_trajectories(self):
        self._rollout_trajectories = wp.zeros((self.env.action_dim, self.num_threads), dtype=wp.float32)

    def sample_controls(self, nominal_traj, num_steps, num_threads, noise_scale=noise_scale):
        controls = wp.zeros((self.env.action_dim, num_threads), dtype=wp.float32)
        for i in range(num_threads):
            controls[:, i] = sample_gaussian(nominal_traj[:, i], noise_scale * nominal_traj[:, i])
        return controls
        



if __name__ == "__main__":
    from env.env_cartpole import CartpoleEnvironment
    
    mpc = Controller(CartpoleEnvironment)
    mpc.run()
