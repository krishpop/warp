import unittest
import os
import sys
import warp as wp
import warp.sim

from warp.tests.test_base import *

sys.path.append(os.path.join(os.path.dirname(warp.__file__), "..", "examples", "env"))

from environment import RenderMode, IntegratorType

import numpy as np
import warp as wp

# default test mode (see get_test_devices())
#   "basic" - only run on CPU and first GPU device
#   "unique" - run on CPU and all unique GPU arches
#   "all" - run on all devices
test_mode = "basic"


def get_test_devices(mode=None):
    if mode is None:
        global test_mode
        mode = test_mode

    devices = []

    # only run on CPU and first GPU device
    if mode == "basic":
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))

    # run on CPU and all unique GPU arches
    elif mode == "unique":
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))

        cuda_devices = wp.get_cuda_devices()

        unique_cuda_devices = {}
        for d in cuda_devices:
            if d.arch not in unique_cuda_devices:
                unique_cuda_devices[d.arch] = d

        devices.extend(list(unique_cuda_devices.values()))

    # run on all devices
    elif mode == "all":
        devices = wp.get_devices()

    return devices


def run_env(env):
    if env.use_graph_capture:
        # create update graph
        wp.capture_begin()
        env.update()
        graph = wp.capture_end()

    q_history = []
    q_history.append(env.state.body_q.numpy().copy().reshape(env.num_envs, -1))
    qd_history = []
    qd_history.append(env.state.body_qd.numpy().copy().reshape(env.num_envs, -1))
    delta_history = []
    if env.integrator_type == IntegratorType.XPBD and env.requires_grad:
        delta_history.append(env.state.body_deltas.numpy().copy().reshape(env.num_envs, -1))
    num_con_history = []
    num_con_history.append(env.model.rigid_contact_inv_weight.numpy().copy().reshape(env.num_envs, -1))

    joint_q_history = []
    joint_q = wp.zeros_like(env.model.joint_q)
    joint_qd = wp.zeros_like(env.model.joint_qd)

    # simulate
    with wp.ScopedTimer("simulate", detailed=False, print=False, active=True):
        running = True
        while running:
            for f in range(env.episode_frames):
                if env.model.particle_count > 1:
                    env.model.particle_grid.build(
                        env.state.particle_q,
                        env.model.particle_max_radius * 2.0,
                    )
                if env.use_graph_capture:
                    wp.capture_launch(graph)
                    env.sim_time += env.frame_dt
                    env.sim_step += env.sim_substeps
                elif not env.requires_grad or env.sim_step < env.sim_steps:
                    env.update()

                q_history.append(env.state.body_q.numpy().copy().reshape(env.num_envs, -1))
                qd_history.append(env.state.body_qd.numpy().copy().reshape(env.num_envs, -1))
                if env.integrator_type == IntegratorType.XPBD and env.requires_grad:
                    delta_history.append(env.state.body_deltas.numpy().copy().reshape(env.num_envs, -1))
                num_con_history.append(env.model.rigid_contact_inv_weight.numpy().copy().reshape(env.num_envs, -1))

                wp.sim.eval_ik(env.model, env.state, joint_q, joint_qd)
                joint_q_history.append(joint_q.numpy().copy().reshape(env.num_envs, -1))

                env.render()
                if env.render_mode == RenderMode.OPENGL and env.renderer.has_exit:
                    running = False
                    break

            if not env.continuous_opengl_render or env.render_mode != RenderMode.OPENGL:
                break

        wp.synchronize()
    env.after_simulate()
    return q_history, qd_history, delta_history, num_con_history, joint_q_history


def check_histories_equal(test, history1, history2, history_name):
    for i in range(len(history1)):
        test.assertTrue(
            np.allclose(history1[i], history2[i]),
            f"{history_name} mismatch at frame {i}, delta={np.max(np.abs(history1[i] - history2[i]))}",
        )


def test_cartpole_single_env_determinism(test, device):
    from env_cartpole import CartpoleEnvironment

    wp.set_device(device)

    CartpoleEnvironment.num_envs = 1
    CartpoleEnvironment.render_mode = RenderMode.NONE
    CartpoleEnvironment.episode_frames = 5  # at 60 fps, 5 frames is 1/12th of a second
    demo = CartpoleEnvironment()

    demo.parse_args()
    demo.init()

    q_history, qd_history, delta_history, num_con_history, joint_q_history = run_env(demo)

    # re-run simulation
    q_history2, qd_history2, delta_history2, num_con_history2, joint_q_history2 = run_env(demo)
    check_histories_equal(test, q_history, q_history2, "q_history")
    check_histories_equal(test, qd_history, qd_history2, "qd_history")
    if len(delta_history) > 0:
        check_histories_equal(test, delta_history, delta_history2, "delta_history")
    check_histories_equal(test, num_con_history, num_con_history2, "num_con_history")
    check_histories_equal(test, joint_q_history, joint_q_history2, "joint_q_history")


def test_cartpole_parallel_env_determinism(test, device):
    from env_cartpole import CartpoleEnvironment

    wp.set_device(device)

    CartpoleEnvironment.num_envs = 2
    CartpoleEnvironment.render_mode = RenderMode.NONE
    CartpoleEnvironment.episode_frames = 5  # at 60 fps, 5 frames is 1/12th of a second
    demo = CartpoleEnvironment()
    demo.parse_args()
    demo.init()

    q_history, qd_history, delta_history, num_con_history, joint_q_history = run_env(demo)
    # check that all q, qd, delta, num_con_history, joint_q_history are same along env dimension
    check_histories_equal(test, [h[0] for h in q_history], [h[1] for h in q_history], "q_history")
    check_histories_equal(test, [h[0] for h in qd_history], [h[1] for h in qd_history], "qd_history")
    if len(delta_history) > 0:
        check_histories_equal(test, [h[0] for h in delta_history], [h[1] for h in delta_history], "delta_history")
    check_histories_equal(test, [h[0] for h in num_con_history], [h[1] for h in num_con_history], "num_con_history")
    check_histories_equal(test, [h[0] for h in joint_q_history], [h[1] for h in joint_q_history], "joint_q_history")


def test_ant_single_env_determinism(test, device):
    from env_ant import AntEnvironment

    wp.set_device(device)

    AntEnvironment.num_envs = 1
    AntEnvironment.render_mode = RenderMode.NONE
    AntEnvironment.episode_frames = 5  # at 60 fps, 5 frames is 1/12th of a second
    demo = AntEnvironment()
    demo.parse_args()
    demo.init()

    q_history, qd_history, delta_history, num_con_history, joint_q_history = run_env(demo)

    # re-run simulation
    q_history2, qd_history2, delta_history2, num_con_history2, joint_q_history2 = run_env(demo)

    # check all q, qd, delta, num_con_history, joint_q_history are same across two runs
    check_histories_equal(test, q_history, q_history2, "q_history")
    check_histories_equal(test, qd_history, qd_history2, "qd_history")
    if len(delta_history) > 0:
        check_histories_equal(test, delta_history, delta_history2, "delta_history")
    check_histories_equal(test, num_con_history, num_con_history2, "num_con_history")
    check_histories_equal(test, joint_q_history, joint_q_history2, "joint_q_history")


def test_ant_parallel_env_determinism(test, device):
    from env_ant import AntEnvironment

    wp.set_device(device)

    AntEnvironment.num_envs = 2
    AntEnvironment.render_mode = RenderMode.NONE
    AntEnvironment.episode_frames = 5  # at 60 fps, 5 frames is 1/12th of a second
    demo = AntEnvironment()
    demo.parse_args()
    demo.init()

    q_history, qd_history, delta_history, num_con_history, joint_q_history = run_env(demo)
    # check that all q, qd, delta, num_con_history, joint_q_history are same along env dimension
    check_histories_equal(test, [h[0] for h in q_history], [h[1] for h in q_history], "q_history")
    check_histories_equal(test, [h[0] for h in qd_history], [h[1] for h in qd_history], "qd_history")
    if len(delta_history) > 0:
        check_histories_equal(test, [h[0] for h in delta_history], [h[1] for h in delta_history], "delta_history")
    check_histories_equal(test, [h[0] for h in num_con_history], [h[1] for h in num_con_history], "num_con_history")
    # check_histories_equal(test, [h[0] for h in joint_q_history], [h[1] for h in joint_q_history], "joint_q_history")

def register(parent):
    devices = get_test_devices()

    class TestDeterministicSim(parent):
        pass

    # add tests for cartpole and ant
    add_function_test(
        TestDeterministicSim,
        "test_cartpole_single_env_determinism",
        test_cartpole_single_env_determinism,
        devices=devices,
    )
    add_function_test(
        TestDeterministicSim,
        "test_cartpole_parallel_env_determinism",
        test_cartpole_parallel_env_determinism,
        devices=devices,
    )

    add_function_test(
        TestDeterministicSim, "test_ant_single_env_determinism", test_ant_single_env_determinism, devices=devices
    )
    add_function_test(
        TestDeterministicSim, "test_ant_parallel_env_determinism", test_ant_parallel_env_determinism, devices=devices
    )

    return TestDeterministicSim


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
