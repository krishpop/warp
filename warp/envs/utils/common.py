import os
import os.path as osp
import torch
import numpy as np
import random
import warp as wp
import warp.sim
import matplotlib.pyplot as plt

from tqdm import trange
from scipy.spatial.transform import Rotation as R
from typing import Optional, List
from inspect import getmembers
from enum import Enum
from dataclasses import dataclass
from .rewards import l1_dist

from warp.sim.model import State


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def seeding(seed=0, torch_deterministic=False):
    """Set seeding with option for torch deterministic with cuda devices."""
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def gen_unique_filename(name, max_id=10):
    id = 0
    pref, ext = osp.splitext(name)
    fn = lambda: f"{pref}-{id}{ext}"
    while osp.exists(fn()) and id < max_id:
        id += 1
    assert id <= max_id, "Too many files with same name, attempted to save as {}".format(fn())
    return fn()


def parse_mesh(stl_file):
    import trimesh
    import numpy as np
    from warp.sim.model import Mesh

    faces = []
    vertices = []

    # handle stl relative paths
    m = trimesh.load(stl_file)

    for v in m.vertices:
        vertices.append(np.array(v))

    for f in m.faces:
        faces.append(int(f[0]))
        faces.append(int(f[1]))
        faces.append(int(f[2]))
    return Mesh(vertices, faces), m.scale


def safe_save(data, path):
    if not osp.exists("data"):
        os.makedirs("data")
    path = gen_unique_filename(f"data/{path}")
    if isinstance(data, dict):
        np.savez(path, **data)
    else:
        np.save(path, data)


def to_torch(arr, device):
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def to_numpy(input):
    if isinstance(input, np.ndarray):
        return input
    return input.detach().cpu().numpy()


def to_warp(input, dtype=float, device="cuda"):
    if isinstance(input, np.ndarray):
        return wp.array(input, dtype=dtype, device=device)
    if isinstance(input, torch.Tensor):
        return wp.from_torch(input, dtype=dtype, device=device)
    if isinstance(input, wp.array):
        return input


class ActionType(Enum):
    POSITION = 0
    TORQUE = 1
    VARIABLE_STIFFNESS = 2


class GoalType(Enum):
    POSITION = 0
    ORIENTATION = 1
    POSE = 2
    TRAJECTORY_POSITION = 3
    TRAJECTORY_ORIENTATION = 4
    TRAJECTORY_POSITION_FORCE = 5
    TRAJECTORY_ORIENTATION_TORQUE = 6
    TRAJECTORY_POSE_WRENCH = 7


class RewardType(Enum):
    DELTA = 0
    EXP = 1
    L2 = 2


# Add 3d equivalents for goal type/create 2d or 3d boolean

POSITION_GOAL_TYPES = [
    GoalType.POSITION,
    GoalType.TRAJECTORY_POSITION_FORCE,
    GoalType.TRAJECTORY_POSITION,
    GoalType.TRAJECTORY_POSE_WRENCH,
]

ORIENTATION_GOAL_TYPES = [
    GoalType.ORIENTATION,
    GoalType.TRAJECTORY_ORIENTATION,
    GoalType.TRAJECTORY_ORIENTATION_TORQUE,
]


class HandType(Enum):
    ALLEGRO = 0
    SHADOW = 1


HAND_ACT_COUNT = {
    (HandType.ALLEGRO, ActionType.POSITION): 16,
    (HandType.ALLEGRO, ActionType.TORQUE): 16,
    (HandType.ALLEGRO, ActionType.VARIABLE_STIFFNESS): 32,
    (HandType.SHADOW, ActionType.POSITION): 24,
    (HandType.SHADOW, ActionType.TORQUE): 24,
}


class ObjectType(Enum):
    CYLINDER_MESH = 0
    CUBE_MESH = 1
    OCTPRISM_MESH = 2
    ELLIPSOID_MESH = 3
    RECTANGULAR_PRISM_FLAT = 4
    RECTANGULAR_PRISM = 5
    SPHERE = 6
    CUBE = 7
    OCTPRISM = 8
    TCDM_STAPLER = 9
    SPRAY_BOTTLE = 10
    PILL_BOTTLE = 11
    BOTTLE = 12
    PLIERS = 13
    SCISSORS = 14
    DISPENSER = 15
    EYEGLASSES = 16
    FAUCET = 17
    STAPLER = 18
    SWITCH = 19
    USB = 20
    REPOSE_CUBE = 21


@dataclass
class GraspParams:
    xform: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    hand_type: HandType = HandType.ALLEGRO
    joint_pos: np.ndarray = None
    stiffness: Optional[float] = None
    damping: Optional[float] = None


SHADOW_HAND_JOINTS = [
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
]


def parse_grasp_data(grasp):
    if len(grasp["qpos"]) == 22:
        hand_type = HandType.ALLEGRO
        qpos = np.array([grasp["qpos"][f"joint_{i}.0"] for i in range(16)])
    elif len(grasp["qpos"]) == 28:
        hand_type = HandType.SHADOW
        qpos = np.array([grasp["qpos"][f"robot0:{joint_name}"] for joint_name in SHADOW_HAND_JOINTS])
    else:
        raise NotImplementedError("Hand type not supported")

    pose_r = [grasp["qpos"][f"WRJR{d}"] for d in "xyz"]
    pose_t = [grasp["qpos"][f"WRJT{d}"] for d in "xyz"]
    pose_r = R.from_euler("xyz", pose_r).as_quat()
    return np.concatenate([pose_t, pose_r]), qpos, hand_type


def load_grasps_npy(path) -> List[GraspParams]:
    data = np.load(path, allow_pickle=True)

    params = []
    for grasp in data:
        hand_pose, qpos, hand_type = parse_grasp_data(grasp)
        params.append(GraspParams(hand_type=hand_type, xform=hand_pose, joint_pos=qpos))
    return params


def clear_state_grads(state: State):
    for k, v in getmembers(state):
        if isinstance(v, wp.array) and v.requires_grad and v.grad is not None:
            v.grad.zero_()
    return state


joint_coord_map = {
    wp.sim.JOINT_PRISMATIC: 1,
    wp.sim.JOINT_REVOLUTE: 1,
    wp.sim.JOINT_BALL: 4,
    wp.sim.JOINT_FREE: 7,
    wp.sim.JOINT_FIXED: 0,
    wp.sim.JOINT_UNIVERSAL: 2,
    wp.sim.JOINT_COMPOUND: 3,
    wp.sim.JOINT_D6: 3,
    wp.sim.JOINT_DISTANCE: 0,
}

supported_joint_types = {
    ActionType.POSITION: [
        wp.sim.JOINT_PRISMATIC,
        wp.sim.JOINT_REVOLUTE,
        wp.sim.JOINT_BALL,
    ],
    ActionType.TORQUE: [wp.sim.JOINT_PRISMATIC, wp.sim.JOINT_REVOLUTE, wp.sim.JOINT_BALL],
}


def profile(env):
    env_count = 2
    env_times = []
    env_size = []

    for i in range(15):
        env.num_environments = env_count
        env.init_sim()
        steps_per_second = env.run()

        env_size.append(env_count)
        env_times.append(steps_per_second)

        env_count *= 2

    # dump times
    for i in range(len(env_times)):
        print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

    # plot
    plt.figure(1)
    plt.plot(env_size, env_times)
    plt.xscale("log")
    plt.xlabel("Number of Envs")
    plt.yscale("log")
    plt.ylabel("Steps/Second")
    plt.show()


def run_env(env, pi=None, num_steps=50, log_runs=False):
    if pi is None:
        # env.reset()
        joint_target_indices = env.env_joint_target_indices

        upper = env.model.joint_limit_upper.numpy().reshape(env.num_envs, -1)[0, joint_target_indices]
        lower = env.model.joint_limit_lower.numpy().reshape(env.num_envs, -1)[0, joint_target_indices]
        joint_start = env.start_joint_q.cpu().numpy()[:, joint_target_indices]

        n_dof = env.num_acts

        # for each degree of freedom, collect a rollout controlling joint target

        def pi(obs, t):
            del obs
            joint_q_targets = (
                np.sin(np.linspace(0, 3 * np.pi, num_steps))[:, None] * (upper - lower) / 2 + (upper + lower) / 2
            )
            action = joint_start.copy()
            action[:, :] = joint_q_targets[t % num_steps]
            return torch.tensor(action, device=str(env.device))

    num_steps = num_steps * n_dof
    actions, states, rewards, _ = collect_rollout(env, num_steps, pi)
    if log_runs:
        np.savez(
            f"{env.env_name}_rollout",
            actions=np.asarray(actions),
            states=np.asarray(states),
            rewards=np.asarray(rewards),
        )


def collect_rollout(env, n_steps, pi, loss_fn=None, plot_body_coords=False, plot_joint_coords=False):
    o = env.reset()
    net_cost = 0.0
    states = []
    actions = []
    rewards = []
    if plot_body_coords:
        q_history, qd_history, delta_history, num_con_history = [], [], [], []
        q_history = []
        q_history.append(env.state_0.body_q.numpy().copy())
        qd_history = []
        qd_history.append(env.state_0.body_qd.numpy().copy())
        delta_history = []
        delta_history.append(env.state_0.body_deltas.numpy().copy())
        num_con_history = []
        num_con_history.append(env.model.rigid_contact_inv_weight.numpy().copy())
    if plot_joint_coords:
        joint_q_history = []

    with trange(n_steps, desc=f"cost={net_cost:.2f}") as pbar:
        for t in pbar:
            ac = pi(o, t)
            actions.append(ac.cpu().detach().numpy())
            o, rew, _, info = env.step(ac)
            if plot_body_coords:
                q_history.append(env.state_0.body_q.numpy().copy())
                qd_history.append(env.state_0.body_qd.numpy().copy())
                delta_history.append(env.state_0.body_deltas.numpy().copy())
                num_con_history.append(env.model.rigid_contact_inv_weight.numpy().copy())
            if plot_joint_coords:
                joint_q_history.append(env.state_0.joint_q.numpy().copy())

            rew = rew.sum().cpu().detach().item()
            if loss_fn is not None:
                loss_fn()

            net_cost += rew
            pbar.set_description(f"cost={net_cost:.2f}, body_f_max={info['body_f_max']:.2f}")
            states.append(o.cpu().detach().numpy())
            rewards.append(rew)
    history = {}
    if plot_body_coords:
        history["q"] = np.stack(q_history)
        history["qd"] = np.stack(qd_history)
        history["delta"] = np.stack(delta_history)
        history["num_con"] = np.stack(num_con_history)
    if plot_joint_coords:
        history["joint_q"] = np.stack(joint_q_history)

    return actions, states, rewards, history
