# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %autosave 0
# %load_ext autoreload
# %autoreload 2

# %%
import warp as wp
import os
import warp.sim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R
from localscope import localscope
from warp.envs.utils import builder as bu
from warp.envs.hand_env import HandObjectTask
from warp.envs.utils.common import HandType, ObjectType, ActionType, GraspParams, load_grasps_npy
from pyglet.math import Vec3 as PyVec3

wp.init()

# %%
object_type = 'spray_bottle'
object_id = None

if object_id is not None:
    grasp_file = f"{object_type}_{object_id}_merged.npy"
else:
    grasp_file = f"{object_type}_merged.npy"
grasp_dir = "/scr-ssd/ksrini/tyler-DexGraspNet/grasp_generation/graspdata_allegro/"
grasps = load_grasps_npy(os.path.join(grasp_dir, grasp_file))

env = HandObjectTask(1, 1, 100, object_type=ObjectType[object_type.upper()])
obs = env.reset()

# %%
env.load_camera_params()
plt.imshow(env.render("rgb_array"))


# %%
@localscope.mfc
def plot_grasp_xform(grasp: GraspParams):
    # Original quaternion
    pos = grasp.xform[:3]
    quat = grasp.xform[3:]

    # Rotation matrix
    rot_mat = R.from_quat(quat).as_matrix()

    # Axes vectors
    # x_axis = rot_mat[:, 0]
    # y_axis = rot_mat[:, 1]
    # z_axis = rot_mat[:, 2]

    x_axis = np.array([0.533986, -0.627104, 0.56728])
    y_axis = np.array([0.779297, 0.595503, -0.1979])
    z_axis = np.array([-0.324763, 0.502729, 0.801218])

    # Create a 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=[pos[0], quat[0]],
            y=[pos[1], quat[1]],
            z=[pos[2], quat[2]],
            name='Original Quaternion',
            marker=dict(size=5, color='orange')
        ),
        go.Scatter3d(
            x=[pos[0], x_axis[0]],
            y=[pos[1], x_axis[1]],
            z=[pos[2], x_axis[2]],
            name='X-Axis',
            marker=dict(size=5, color='green')
        ),
        go.Scatter3d(
            x=[pos[0], y_axis[0]],
            y=[pos[1], y_axis[1]],
            z=[pos[2], y_axis[2]],
            name='Y-Axis',
            marker=dict(size=5, color='blue')
        ),
        go.Scatter3d(
            x=[pos[0], z_axis[0]],
            y=[pos[1], z_axis[1]],
            z=[pos[2], z_axis[2]],
            name='Z-Axis',
            marker=dict(size=5, color='red')
        ),
        go.Scatter3d(
            x=[pos[0], 1],
            y=[pos[1], 0],
            z=[pos[2], 0],
            name='Original X-Axis',
            marker=dict(size=2, color='green')
        ),
        go.Scatter3d(
            x=[pos[0], 0],
            y=[pos[1], 1],
            z=[pos[2], 0],
            name='Original Y-Axis',
            marker=dict(size=2, color='blue')
        ),
        go.Scatter3d(
            x=[pos[0], 0],
            y=[pos[1], 0],
            z=[pos[2], 1],
            name='Original Z-Axis',
            marker=dict(size=2, color='red')
        ),
    ])

    # Set the layout
    fig.update_layout(
        title="Quaternion and Axes Vectors",
        scene=dict(
            xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ),
    showlegend=True
    )

    # Show the plot
    fig.show()

# %%
env.calculateObservations().shape

# %%
plot_grasp_xform(grasps[0])

# %%
env.render()
env.renderer._camera_pos = PyVec3(0.0, .5, 2.0)
env.renderer.update_view_matrix()
env.render()

# %%
env.renderer.

# %%
pos, q = grasps[0].xform[:3], grasps[0].xform[3:]

# %%
# %matplotlib widget

# %%
while True:
    env.renderer.axis_instancer.allocate_instances(
        positions=[pos],
        rotations=,
        colors1=,
        colors2=)
    env.render()

# %%
env.env_joint_mask

# %%
builder = wp.sim.ModelBuilder()
bu.create_allegro_hand(
    builder,
    ActionType.POSITION,
    stiffness=1000,
    damping=0.1,
    base_joint=None,
    floating_base=True,
    # base_joint="rx, ry, rz",
    hand_start_position=(0,0.4,0.),
    hand_start_orientation=(0.,0.,0.),
)

# %%
model = builder.finalize()

# %%
np.concatenate([grasps[0].xform, grasps[0].joint_pos]).size

# %%
state = model.state()
state.joint_q.assign(np.concatenate([grasps[0].xform, grasps[0].joint_pos]))

# %%
print(state.joint_q)

# %%
state = model.state()

wp.sim.eval_fk(model, state.joint_q, state.joint_qd, None, state)

# %%
renderer = wp.sim.render.SimRendererOpenGL(
    model,
    "env",
    up_axis="y",
    show_joints=True
)

# %%
renderer.show_joints= False
renderer.draw_axis = False

# %%
while True:
    renderer.begin_frame(0.0)
    renderer.render(state)
    renderer.end_frame()

# %%
print(state.joint_q)

# %%
HandObjectTask.usd_render_settings['show_joints'] = True
env = HandObjectTask(1,1,100, hand_type=HandType["SHADOW"])

obs = env.reset()
env.render()
env.renderer._camera_pos = PyVec3(0.0, .5, 2.0)
env.renderer.update_view_matrix()
env.render()

# %%
while True:
    env.render()

# %%
from common import load_grasps_npy

# %%
grasps = load_grasps_npy("/scr-ssd/ksrini/tyler-DexGraspNet/grasp_generation/graspdata_allegro/spray_bottle_merged.npy")

# %%
env.reset_buf != 0

# %%
env.hand_joint_start = 0

# %%
env.model.body_name

# %%
env.model.joint_X_p

# %%
env._set_hand_base_xform(env.reset_buf != 0, torch.tensor(grasps[0].xform, dtype=torch.float32, device="cuda"))

# %%
from rewards import l1_dist

# %%
env.rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}

# %%
env.renderer.show_joints = True
env.render()

# %%
env.step(torch.tensor(env.action_space.sample()*0, dtype=torch.float32, device='cuda'))

# %%
env.rew_params

# %%
env.render()
