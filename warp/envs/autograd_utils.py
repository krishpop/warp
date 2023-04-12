import torch
import warp as wp
import numpy as np
from typing import Union
from inspect import getmembers
from warp.sim.model import State, Model
from torch.cuda.amp import custom_fwd, custom_bwd

# from warp.tests.grad_utils import check_kernel_jacobian, check_backward_pass, plot_jacobian_comparison


def clear_grads(obj: Union[State, Model], filter=None):
    for k, v in getmembers(obj):
        if filter is not None and not filter(k):
            continue
        if isinstance(v, wp.array) and v.requires_grad and v.grad is not None:
            v.grad.zero_()
    return obj


@wp.kernel
def assign_act_kernel(
    act: wp.array(dtype=float),  # unflattened shape (n, 4)
    num_acts: int,
    dof_count: int,
    q_offset: int,
    # outputs
    joint_act: wp.array(dtype=float),  # unflattened shape (n, 6)
):
    i, j = wp.tid()
    joint_act_idx = i * dof_count + j + q_offset  # skip object joint
    act_idx = i * num_acts + j
    wp.atomic_add(joint_act, joint_act_idx, act[act_idx])


def assign_act(
    act,
    joint_act,
    num_acts,
    num_envs,
    dof_count,
    q_offset,
):
    assert (
        np.prod(act.shape) == num_envs * num_acts
    ), f"act shape {act.shape} is not {num_envs} x {num_acts}"
    act_count = num_acts
    wp.launch(
        kernel=assign_act_kernel,
        dim=(num_envs, act_count),
        device=joint_act.device,
        inputs=[act, num_acts, dof_count, q_offset],
        outputs=[joint_act],
    )
    return


def get_compute_graph(func, kwargs={}, tape=None):
    if tape is not None:
        with tape:
            wp.capture_begin()
            func(**kwargs)
            return wp.capture_end()
    wp.capture_begin()
    func(**kwargs)
    return wp.capture_end()


def forward_simulate(ctx, forward=False, requires_grad=False):
    joint_q_end = ctx.graph_capture_params["joint_q_end"]
    joint_qd_end = ctx.graph_capture_params["joint_qd_end"]
    if forward or not ctx.capture_graph:
        model = ctx.model
        state_temp = ctx.state_in
        state_out = ctx.state_out
    else:
        model = ctx.backward_model
        state_temp = ctx.bwd_state_in
        state_out = ctx.bwd_state_out

    num_envs = ctx.act_params["num_envs"]
    dof_count = ctx.act_params["dof_count"]
    q_offset = ctx.act_params["q_offset"]
    num_acts = ctx.act_params["num_acts"]  # ctx.act.size // num_envs

    assign_act(
        ctx.act,
        ctx.joint_act,
        num_acts=num_acts,
        num_envs=num_envs,
        dof_count=dof_count,
        q_offset=q_offset,
    )
    for step in range(ctx.substeps):
        state_in = state_temp
        state_in.clear_forces()
        # if step < ctx.substeps - 1:
        #     # create new states to preserve gradients
        #     if not ctx.capture_graph:
        #         state_temp = model.state(requires_grad=requires_grad)
        #     else:
        #         state_temp = ctx.state_out
        # else:
        if forward or step == ctx.substeps - 1:
            state_temp = state_out
        else:
            state_temp = ctx.state_list[step]
        if model.ground:
            if not forward:
                model.allocate_rigid_contacts()
            wp.sim.collide(model, state_in)
        state_temp = ctx.integrator.simulate(
            model,
            state_in,
            state_temp,
            ctx.dt,
        )
    wp.sim.eval_ik(ctx.model, state_out, joint_q_end, joint_qd_end)


class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        simulate_params,
        graph_capture_params,
        act_params,
        action,
        body_q,
        body_qd,
    ):
        ctx.model = simulate_params["model"]
        ctx.backward_model = graph_capture_params["bwd_model"]
        ctx.integrator = simulate_params["integrator"]
        ctx.dt, ctx.substeps = simulate_params["dt"], simulate_params["substeps"]
        ctx.state_in = simulate_params["state_in"]
        ctx.state_out = simulate_params["state_out"]
        ctx.state_list = simulate_params.get("state_list", None)
        ctx.return_body = simulate_params.get("ag_return_body", True)
        ctx.act_params = act_params
        ctx.act = act_params["act"]
        ctx.joint_act = act_params["joint_act"]
        ctx.act_pt = action
        ctx.act.assign(wp.from_torch(ctx.act_pt.detach()))
        ctx.body_q_pt = body_q
        ctx.body_qd_pt = body_qd
        ctx.joint_q_end = graph_capture_params["joint_q_end"]
        ctx.joint_qd_end = graph_capture_params["joint_qd_end"]
        ctx.capture_graph = graph_capture_params["capture_graph"]
        ctx.graph_capture_params = graph_capture_params

        # if using graph capture, need to assign states as in/out buffers cannot be swapped
        if ctx.capture_graph:
            ctx.state_in.body_q.assign(wp.from_torch(ctx.body_q_pt, dtype=wp.transform))
            ctx.state_in.body_qd.assign(
                wp.from_torch(ctx.body_qd_pt, dtype=wp.spatial_vector)
            )

        # record gradients for act, joint_q, and joint_qd
        ctx.act.requires_grad = True
        ctx.joint_q_end.requires_grad = True
        ctx.joint_qd_end.requires_grad = True
        ctx.state_in.body_q.requires_grad = True
        ctx.state_in.body_qd.requires_grad = True
        ctx.state_out.body_q.requires_grad = True
        ctx.state_out.body_qd.requires_grad = True

        if ctx.capture_graph:
            ctx.tape = graph_capture_params["tape"]
            ctx.bwd_state_in = simulate_params["bwd_state_in"]
            ctx.bwd_state_out = simulate_params["bwd_state_out"]
            ctx.bwd_state_in.body_q.requires_grad = True
            ctx.bwd_state_in.body_qd.requires_grad = True
            ctx.bwd_state_out.body_q.requires_grad = True
            ctx.bwd_state_out.body_qd.requires_grad = True
            ctx.forward_graph = graph_capture_params.get(
                "forward_graph",
                get_compute_graph(forward_simulate, {"ctx": ctx, "forward": True}),
            )
            graph_capture_params["forward_graph"] = ctx.forward_graph
            wp.capture_launch(ctx.forward_graph)
        else:
            ctx.tape = wp.Tape()
            with ctx.tape:
                forward_simulate(ctx, forward=False, requires_grad=True)

        joint_q_end = wp.to_torch(ctx.graph_capture_params["joint_q_end"])
        joint_qd_end = wp.to_torch(ctx.graph_capture_params["joint_qd_end"])
        if ctx.return_body:
            body_q, body_qd = wp.to_torch(ctx.state_out.body_q), wp.to_torch(
                ctx.state_out.body_qd
            )
            return (joint_q_end, joint_qd_end, body_q, body_qd)
        else:
            return (joint_q_end, joint_qd_end)

    @staticmethod
    @custom_bwd
    def backward(ctx, *adj_arrs):
        if ctx.return_body:
            adj_joint_q, adj_joint_qd, adj_body_q, adj_body_qd = adj_arrs
        else:
            adj_joint_q, adj_joint_qd = adj_arrs
            adj_body_q, adj_body_qd = None, None
        # def backward(ctx, adj_joint_q, adj_joint_qd, adj_body_q, adj_body_qd):
        # map incoming Torch grads to our output variables
        joint_q_end = ctx.graph_capture_params["joint_q_end"]
        joint_qd_end = ctx.graph_capture_params["joint_qd_end"]
        state_in = ctx.state_in
        state_out = ctx.state_out
        tape = ctx.tape
        if ctx.capture_graph:
            state_in = ctx.bwd_state_in
            state_out = ctx.bwd_state_out
            state_in.body_q.assign(wp.from_torch(ctx.body_q_pt, dtype=wp.transform))
            state_in.body_qd.assign(
                wp.from_torch(ctx.body_qd_pt, dtype=wp.spatial_vector)
            )
            ctx.act.zero_()
            ctx.act.grad.zero_()
            ctx.act.assign(wp.from_torch(ctx.act_pt.detach()))
            assert ctx.act.grad.numpy().sum() == 0
            assert state_in.body_q.grad.numpy().sum() == 0
            assert state_out.body_q.grad.numpy().sum() == 0
            # Do forward sim again, allocating rigid pairs and intermediate states
            # ctx.graph_capture_params[
            #     "bwd_forward_graph"
            # ] = ctx.graph_capture_params.get(
            #     "bwd_forward_graph",
            #     get_compute_graph(
            #         forward_simulate,
            #         {"ctx": ctx, "requires_grad": True, "forward": False},
            #         tape,
            #     ),
            # )
            # wp.capture_launch(ctx.graph_capture_params["bwd_forward_graph"])
            with tape:  # check if graph capture works for this
                forward_simulate(ctx, forward=False, requires_grad=True)

        joint_q_end.grad.assign(wp.from_torch(adj_joint_q))
        joint_qd_end.grad.assign(wp.from_torch(adj_joint_qd))
        if ctx.return_body:
            if state_out.body_q.grad is None:
                state_out.body_q.grad = wp.zeros_like(adj_body_q, dtype=wp.transform)
                state_out.body_qd.grad = wp.zeros_like(
                    adj_body_qd, dtype=wp.spatial_vector
                )
            state_out.body_q.grad.assign(wp.from_torch(adj_body_q, dtype=wp.transform))
            state_out.body_qd.grad.assign(
                wp.from_torch(adj_body_qd, dtype=wp.spatial_vector)
            )

        if ctx.capture_graph:
            ctx.graph_capture_params["backward_graph"] = ctx.graph_capture_params.get(
                "backward_graph", get_compute_graph(tape.backward)
            )
            wp.capture_launch(ctx.graph_capture_params["backward_graph"])
        else:
            tape.backward()

        joint_act_grad = wp.to_torch(tape.gradients[ctx.act]).clone()
        # Unnecessary copying of grads, grads should already be recorded by context
        body_q_grad = wp.to_torch(tape.gradients[state_in.body_q]).clone()
        body_qd_grad = wp.to_torch(tape.gradients[state_in.body_qd]).clone()

        tape.zero()
        # return adjoint w.r.t. inputs
        return (
            None,  # simulate_params,
            None,  # graph_capture_params,
            None,  # act_params,
            joint_act_grad,  # action,
            body_q_grad,  # body_q,
            body_qd_grad,  # body_qd,
        )


forward_ag = IntegratorSimulate.apply
