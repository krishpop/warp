import numpy as np
import torch
import warp as wp
import builder_utils as bu
from hand_env import HandObjectTask
from warp.envs.common import run_env, collect_rollout, ObjectType
from environment import RenderMode
from warp.optim import Adam


def main():
    rew_params = {"hand_target_obs_l1_dist": (bu.l1_dist, ["hand_pos", "target_pos"], 1.0)}
    env = HandObjectTask(
        num_envs=1,
        num_obs=0,
        episode_length=1000,
        seed=0,
        no_grad=False,
        render=True,
        stochastic_init=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type=None,
        object_id=0,
        stiffness=5000.0,
        damping=0.5,
        rew_params=rew_params,
    )
    env.reset()
    joint_target_indices = env.env_joint_target_indices

    # upper = env.model.joint_limit_upper.numpy().reshape(env.num_envs, -1)[0, joint_target_indices]
    # lower = env.model.joint_limit_lower.numpy().reshape(env.num_envs, -1)[0, joint_target_indices]
    joint_start = env.start_joint_q.cpu().numpy()[:, joint_target_indices]
    action = torch.tensor(joint_start, device=str(env.device))
    stiffness = env.hand_stiffness
    damping = env.hand_damping
    stiffness.requires_grad = True
    damping.requires_grad = True
    num_states = 1
    optimizer = Adam([stiffness, damping], lr=1e-3)
    loss = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    num_opt_steps = 100
    pi = lambda x, y: action

    for i in range(num_opt_steps):
        num_steps = 1  # 2 * num_states + 1
        loss.zero_()
        tape = wp.Tape()
        with tape:
            actions, states, rewards = collect_rollout(env, num_steps, pi, loss=loss)
        tape.backward(loss=loss)
        __import__("ipdb").set_trace()
        optimizer.step([stiffness.grad, damping.grad])
        tape.zero()
        # loss = -rewards.sum()
        # loss.backward()
        # optimizer.step()
        np.savez(
            f"{env.env_name}_dof_rollout-{i}",
            actions=np.asarray(actions),
            states=np.asarray(states),
            rewards=np.asarray(rewards),
        )


if __name__ == "__main__":
    main()
