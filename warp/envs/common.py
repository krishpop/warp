import os
import os.path as osp
import torch
import numpy as np
import random
import warp as wp
import warp.sim
import matplotlib.pyplot as plt

from tqdm import trange
from inspect import getmembers
from enum import Enum
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


OBJ_PATHS = {
    ObjectType.CYLINDER_MESH: "cylinder.stl",
    ObjectType.CUBE_MESH: "cube.stl",
    ObjectType.OCTPRISM_MESH: "octprism-2.stl",
    ObjectType.ELLIPSOID_MESH: "ellipsoid.stl",
}

TCDM_MESH_PATHS = {ObjectType.TCDM_STAPLER: "meshes/objects/stapler/stapler.stl"}
TCDM_TRAJ_PATHS = {ObjectType.TCDM_STAPLER: "stapler_lift.npz"}
TCDM_OBJ_NAMES = {ObjectType.TCDM_STAPLER: "stapler-lift"}


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
    wp.sim.JOINT_DISTANCE: 0,
}

supported_joint_types = {
    ActionType.POSITION: [
        wp.sim.JOINT_PRISMATIC,
        wp.sim.JOINT_REVOLUTE,
    ],
    ActionType.TORQUE: [
        wp.sim.JOINT_PRISMATIC,
        wp.sim.JOINT_REVOLUTE,
    ],
}


def run_env(Env, num_states=500):
    env = Env()
    # env.parse_args()
    if env.profile:

        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            env.num_envs = env_count
            env.init()
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
    else:
        # env.reset()
        joint_target_indices = env.env_joint_target_indices

        upper = env.model.joint_limit_upper.numpy().reshape(env.num_envs, -1)[:, joint_target_indices]
        lower = env.model.joint_limit_lower.numpy().reshape(env.num_envs, -1)[:, joint_target_indices]
        joint_start = env.start_joint_q.cpu().numpy()[:, joint_target_indices]

        n_dof = env.num_acts

        for i in range(n_dof):
            joint_q_targets = (
                0.8 * np.sin(np.linspace(0, 3 * np.pi, 2 * num_states + 1)) * (upper[i] - lower[i]) / 2
                + (upper[i] + lower[i]) / 2
            )

            def pi(t):
                action = joint_start.copy()
                action[:, i] = joint_q_targets[t]
                return action

            num_steps = 2 * num_states + 1
            ac, states = collect_states(env, num_steps, pi)
            np.savez(f"run_dof-{i}", ac=np.asarray(ac), states=np.asarray(states))


def collect_states(env, n_steps, pi):
    o = env.reset()
    prev_q = env.extras["object_joint_pos"]
    cost = 0.0
    states = []
    actions = []
    with trange(n_steps, desc=f"cost={cost:.2f}") as pbar:
        for t in pbar:
            ac = pi(t)
            actions.append(ac)
            ac = torch.tensor(ac).to(str(env.device))
            o, rew, _, info = env.step(ac)
            # net_qdelta += torch.abs(info["object_joint_pos"] - prev_q).detach().cpu().numpy().sum().item()
            prev_q = info["object_joint_pos"]
            cost += rew.sum()
            pbar.set_description(f"cost={cost:.2f}")
            states.append(o.cpu().numpy())
    return actions, states
