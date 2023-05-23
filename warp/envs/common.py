import os
import os.path as osp
import torch
import numpy as np
import random
import warp as wp

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
    PLIER = 12
    SCISSORS = 13


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
