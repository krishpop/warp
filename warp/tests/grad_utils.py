# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import defaultdict
from typing import Callable, List, Literal, Tuple
import warp as wp
from warp.context import Devicelike
import numpy as np

class FontColors:
    # https://stackoverflow.com/a/287944
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def check_gradient(func: Callable, func_name: str, inputs: List, device: Devicelike, eps: float = 1e-4, tol: float = 1e-2):
    """
    Checks that the gradient of the Warp kernel is correct by comparing it to the
    numerical gradient computed using finite differences.
    Note that this function only works for kernels with an output scalar array of length 1.
    """

    module = wp.get_module(func.__module__)
    kernel = wp.Kernel(func=func, key=func_name, module=module)

    def f(xs):
        # call the kernel without taping for finite differences
        wp_xs = [
            wp.array(xs[i], ndim=1, dtype=inputs[i].dtype, device=device)
            for i in range(len(inputs))
        ]
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(kernel, dim=1, inputs=wp_xs, outputs=[output], device=device)
        return output.numpy()[0]

    # compute numerical gradient
    numerical_grad = []
    np_xs = []
    for i in range(len(inputs)):
        np_xs.append(inputs[i].numpy().flatten().copy())
        numerical_grad.append(np.zeros_like(np_xs[-1]))
        inputs[i].requires_grad = True

    for i in range(len(np_xs)):
        for j in range(len(np_xs[i])):
            np_xs[i][j] += eps
            y1 = f(np_xs)
            np_xs[i][j] -= 2*eps
            y2 = f(np_xs)
            np_xs[i][j] += eps
            numerical_grad[i][j] = (y1 - y2) / (2*eps)

    # compute analytical gradient
    tape = wp.Tape()
    output = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    with tape:
        wp.launch(kernel, dim=1, inputs=inputs,
                  outputs=[output], device=device)

    tape.backward(loss=output)

    # compare gradients
    for i in range(len(inputs)):
        grad = tape.gradients[inputs[i]]
        assert_np_equal(grad.numpy(), numerical_grad[i], tol=tol)

    tape.zero()


def is_differentiable(x):
    # TODO add support for structs
    return isinstance(x, wp.array) and x.dtype not in (wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64)


def flatten_arrays(xs):
    # flatten arrays that make sense to differentiate
    arrays = []
    for x in xs:
        if isinstance(x, wp.codegen.StructInstance):
            for varname in x._struct_.vars:
                var = getattr(x, varname)
                if is_differentiable(var):
                    arrays.append(var.numpy().flatten())
        if is_differentiable(x):
            arrays.append(x.numpy().flatten())
    return np.concatenate(arrays)


def create_diff_copies(xs, require_grad=True):
    # create copies of arrays that make sense to differentiate
    diffs = []
    for x in xs:
        if isinstance(x, wp.codegen.StructInstance):
            new_struct = type(x)()
            for varname in x._struct_.vars:
                var = getattr(x, varname)
                if is_differentiable(var):
                    dvar = wp.clone(var)
                    dvar.requires_grad = require_grad
                    setattr(new_struct, varname, dvar)
                elif isinstance(var, wp.array):
                    setattr(new_struct, varname, wp.clone(var))
                else:
                    setattr(new_struct, varname, var)
            diffs.append(new_struct)
        elif is_differentiable(x):
            dx = wp.clone(x)
            dx.requires_grad = require_grad
            diffs.append(dx)
        elif isinstance(x, wp.array):
            diffs.append(wp.clone(x))
        else:
            diffs.append(x)
    return diffs


def onehot(dim, i):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v

def get_device(xs):
    for x in xs:
        if isinstance(x, wp.array):
            return x.device
        elif isinstance(x, wp.codegen.StructInstance):
            for varname in x._struct_.vars:
                var = getattr(x, varname)
                if isinstance(var, wp.array):
                    return var.device
    return wp.get_preferred_device()


def kernel_jacobian(kernel: wp.Kernel, dim: int, inputs: List[wp.array], outputs: List[wp.array], max_outputs_per_var=-1):
    """
    Computes the Jacobian of a Warp kernel launch mapping from all differentiable inputs to all differentiable outputs.
    """
    assert len(outputs) > 0, "Must specify at least one output"

    diff_in = flatten_arrays(inputs)
    diff_out = flatten_arrays(outputs)

    num_in = len(diff_in)
    num_out = len(diff_out)

    # compute analytical Jacobian
    jac_ad = np.zeros((num_out, num_in), dtype=np.float32)

    diff_inputs = create_diff_copies(inputs)
    diff_outputs = create_diff_copies(outputs)
    device = get_device(inputs + outputs)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=kernel,
            dim=dim,
            inputs=diff_inputs,
            outputs=diff_outputs,
            device=device)
    tape.zero()

    row_id = 0
    def eval_row(out):
        nonlocal row_id
        nonlocal tape
        nonlocal jac_ad
        np_out = out.numpy()
        out_shape = np_out.shape
        np_out = np_out.flatten()
        limit = min(max_outputs_per_var, len(np_out)) if max_outputs_per_var > 0 else len(np_out)
        for j in range(limit):
            out.grad = wp.array(onehot(len(np_out), j).reshape(out_shape), dtype=out.dtype, device=out.device)
            tape.backward()
            col_id = 0
            for input in diff_inputs:
                # fill in Jacobian columns from input gradients
                if isinstance(input, wp.codegen.StructInstance):
                    for varname in input._struct_.vars:
                        var = getattr(input, varname)
                        if is_differentiable(var):
                            grad = tape.gradients[var].numpy().flatten()
                            jac_ad[row_id, col_id:col_id + len(grad)] = grad
                            col_id += len(grad)
                elif is_differentiable(input):
                    grad = tape.gradients[input].numpy().flatten()
                    jac_ad[row_id, col_id:col_id + len(grad)] = grad
                    col_id += len(grad)
            tape.zero()
            row_id += 1
    for out in diff_outputs:
        # loop over Jacobian rows, select output dimension to differentiate
        if isinstance(out, wp.codegen.StructInstance):
            for varname in out._struct_.vars:
                var = getattr(out, varname)
                if is_differentiable(var):
                    eval_row(var)
        elif is_differentiable(out):
            eval_row(out)

    return jac_ad, flatten_arrays(diff_inputs), flatten_arrays(diff_outputs)

@wp.kernel
def normalize_transforms(xs: wp.array(dtype=wp.transform)):
    tid = wp.tid()
    x = xs[tid]
    xs[tid] = wp.transform(wp.transform_get_translation(x), wp.normalize(wp.transform_get_rotation(x)))

@wp.kernel
def normalize_quats(xs: wp.array(dtype=wp.quat)):
    tid = wp.tid()
    x = xs[tid]
    xs[tid] = wp.normalize(x)

def normalize_inputs(xs: wp.array):
    """
    Normalizes quaternion inputs to ensure the finite difference Jacobian makes sense.
    """
    # if xs.dtype == wp.transform:
    #     wp.launch(normalize_transforms, dim=len(xs), inputs=[xs], device=xs.device)
    # elif xs.dtype == wp.quat:
    #     wp.launch(normalize_quats, dim=len(xs), inputs=[xs], device=xs.device)
    return xs

def kernel_jacobian_fd(kernel: wp.Kernel, dim: int, inputs: List[wp.array], outputs: List[wp.array], eps: float = 1e-4, max_fd_dims_per_var: int = 500):
    """
    Computes the Jacobian of a Warp kernel launch mapping from all differentiable inputs to all differentiable outputs
    using finite differences.
    """
    assert len(outputs) > 0, "Must specify at least one output"

    diff_in = flatten_arrays(inputs)
    diff_out = flatten_arrays(outputs)

    num_in = len(diff_in)
    num_out = len(diff_out)

    diff_inputs = create_diff_copies(inputs, require_grad=False)
    device = get_device(inputs + outputs)

    def f(xs):
        # make copy of output arrays, because some kernels may modify them in-place
        diff_outputs = create_diff_copies(outputs)
        # call the kernel without taping for finite differences
        wp.launch(kernel, dim=dim, inputs=xs, outputs=diff_outputs, device=device)
        return flatten_arrays(diff_outputs)

    # compute numerical Jacobian
    jac_fd = np.zeros((num_out, num_in), dtype=np.float32)

    col_id = 0
    for input_id, input in enumerate(diff_inputs):
        if isinstance(input, wp.codegen.StructInstance):
            for varname in input._struct_.vars:
                var = getattr(input, varname)
                if is_differentiable(var):
                    np_in = var.numpy().copy()
                    np_in_original = np_in.copy()
                    in_shape = np_in.shape
                    np_in = np_in.flatten()
                    limit = min(max_fd_dims_per_var, len(np_in)) if max_fd_dims_per_var > 0 else len(np_in)
                    for j in range(limit):
                        np_in[j] += eps
                        setattr(diff_inputs[input_id], varname, normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)))
                        y1 = f(diff_inputs)
                        np_in[j] -= 2*eps
                        setattr(diff_inputs[input_id], varname, normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)))
                        y2 = f(diff_inputs)
                        setattr(diff_inputs[input_id], varname, wp.array(np_in_original, dtype=var.dtype, device=var.device))
                        jac_fd[:, col_id] = (y1 - y2) / (2*eps)
                        col_id += 1
        elif is_differentiable(input):
            np_in = input.numpy().copy()
            np_in_original = np_in.copy()
            in_shape = np_in.shape
            np_in = np_in.flatten()
            limit = min(max_fd_dims_per_var, len(np_in)) if max_fd_dims_per_var > 0 else len(np_in)
            for j in range(limit):
                np_in[j] += eps
                diff_inputs[input_id] = normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=input.dtype, device=input.device))
                y1 = f(diff_inputs)
                np_in[j] -= 2*eps
                diff_inputs[input_id] = normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=input.dtype, device=input.device))
                y2 = f(diff_inputs)
                diff_inputs[input_id] = wp.array(np_in_original, dtype=input.dtype, device=input.device)
                jac_fd[:, col_id] = (y1 - y2) / (2*eps)
                col_id += 1

    return jac_fd


def plot_jacobian_comparison(
    jac_ad, jac_fd, title="",
    input_ticks=None, input_ticks_labels=None,
    output_ticks=None, output_ticks_labels=None,
    highlight_xs=None, highlight_ys=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(1, 3)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    def plot_matrix(ax, mat, vmin, vmax):
        mat = np.copy(mat)
        mat[mat==0.0] = np.nan
        if vmin is not None and vmin < vmax and vmin != 0.0:
            ax.imshow(np.abs(mat), cmap='jet', interpolation='nearest', norm=LogNorm())
        else:
            ax.imshow(np.abs(mat), cmap='jet', interpolation='nearest')
        if input_ticks is not None:
            ax.set_xticks(input_ticks)
            if input_ticks_labels is not None:
                ax.set_xticklabels([f"{label} ({tick})" for label, tick in zip(input_ticks_labels, input_ticks)], rotation=90)
        if output_ticks is not None:
            ax.set_yticks(output_ticks)
            if output_ticks_labels is not None:
                ax.set_yticklabels([f"{label} ({tick})" for label, tick in zip(output_ticks_labels, output_ticks)])
    vmin = min(np.min(jac_ad), np.min(jac_fd))
    vmax = max(np.max(jac_ad), np.max(jac_fd))
    plot_matrix(axs[0], jac_ad, vmin, vmax)
    axs[0].set_title("Analytical")
    plot_matrix(axs[1], jac_fd, vmin, vmax)
    axs[1].set_title("Finite Difference")
    diff = jac_ad - jac_fd
    plot_matrix(axs[2], diff, None, None)
    axs[2].set_title("Difference")
    if highlight_xs is not None and highlight_ys is not None:
        axs[2].scatter(highlight_xs, highlight_ys, marker='x', color='red')
    plt.tight_layout(h_pad=0.0, w_pad=0.0)
    plt.show()


def check_kernel_jacobian(kernel: Callable, dim: Tuple[int], inputs: List, outputs: List = [], eps: float = 1e-4, max_fd_dims_per_var: int = 500, max_outputs_per_var: int = 500, atol: float = 100.0, rtol: float = 1e-2, plot_jac_on_fail: bool = False, tabulate_errors: bool = True, warn_about_missing_requires_grad: bool = True):
    """
    Checks that the Jacobian of the Warp kernel is correct by comparing it to the
    numerical Jacobian computed using finite differences.
    """

    if warn_about_missing_requires_grad:
        # check that the kernel arguments have requires_grad enabled
        for input_id, input in enumerate(inputs):
            if isinstance(input, wp.codegen.StructInstance):
                for varname in input._struct_.vars:
                    var = getattr(input, varname)
                    if is_differentiable(var) and not var.requires_grad:
                        print(FontColors.WARNING +
                            f"Warning: input \"{kernel.adj.args[input_id].label}.{varname}\" is differentiable but requires_grad is False" + FontColors.ENDC)
            elif is_differentiable(input) and not input.requires_grad:
                print(FontColors.WARNING +
                    f"Warning: input \"{kernel.adj.args[input_id].label}\" is differentiable but requires_grad is False" + FontColors.ENDC)
        for output_id, output in enumerate(outputs):
            if isinstance(output, wp.codegen.StructInstance):
                for varname in output._struct_.vars:
                    var = getattr(output, varname)
                    if is_differentiable(var) and not var.requires_grad:
                        print(FontColors.WARNING +
                            f"Warning: output \"{kernel.adj.args[output_id + len(inputs)].label}.{varname}\" is differentiable but requires_grad is False" + FontColors.ENDC)
            elif is_differentiable(output) and not output.requires_grad:
                print(FontColors.WARNING +
                    f"Warning: output \"{kernel.adj.args[output_id + len(inputs)].label}\" is differentiable but requires_grad is False" + FontColors.ENDC)

    # find input/output names mapping to Jacobian indices for tick labels
    input_ticks_labels = []
    input_ticks = []
    input_lengths = {}
    i = 0
    for id, x in enumerate(inputs):
        name = kernel.adj.args[id].label
        if isinstance(x, wp.codegen.StructInstance):
            for varname in x._struct_.vars:
                var = getattr(x, varname)
                if is_differentiable(var):
                    sname = f"{name}.{varname}"
                    input_ticks_labels.append(sname)
                    input_ticks.append(i)
                    input_lengths[sname] = len(var.numpy().flatten())
                    i += input_lengths[sname]
        elif is_differentiable(x):  
            input_ticks_labels.append(name)
            input_ticks.append(i)
            input_lengths[name] = len(x.numpy().flatten())
            i += input_lengths[name]
    output_ticks_labels = []
    output_ticks = []
    output_lengths = {}
    i = 0
    for id, x in enumerate(outputs):
        name = kernel.adj.args[id + len(inputs)].label
        if isinstance(x, wp.codegen.StructInstance):
            for varname in x._struct_.vars:
                var = getattr(x, varname)
                if is_differentiable(var):
                    sname = f"{name}.{varname}"
                    output_ticks_labels.append(sname)
                    output_ticks.append(i)
                    output_lengths[sname] = len(var.numpy().flatten())
                    i += output_lengths[sname]
        elif is_differentiable(x):
            output_ticks_labels.append(name)
            output_ticks.append(i)
            output_lengths[name] = len(x.numpy().flatten())
            i += output_lengths[name]

    def find_variable_names(idx: Tuple[int]) -> Tuple[str]:
        # idx is the row, column index in the Jacobian, need to find corresponding output, input var names
        output_label = output_ticks_labels[0]
        for i, tick in enumerate(output_ticks[1:]):
            if idx[0] >= tick:
                output_label = output_ticks_labels[i+1]
        input_label = input_ticks_labels[0]
        for i, tick in enumerate(input_ticks[1:]):
            if idx[1] >= tick:
                input_label = input_ticks_labels[i+1]
        return input_label, output_label

    def compute_max_abs_error(a, b):
        abs_diff = np.abs(a - b)
        max_abs_error = np.max(abs_diff)
        max_abs_error_idx = np.unravel_index(
            np.argmax(abs_diff), abs_diff.shape)
        return max_abs_error, max_abs_error_idx

    def compute_max_rel_error(a, b):
        denom = np.abs(a)
        absb = np.abs(b)
        denom[denom < absb] = absb[denom < absb]
        denom[denom == 0.0] = 1.0
        rel_diff = np.abs(a - b) / denom
        rel_diff[np.isnan(rel_diff)] = 0
        max_rel_error = np.max(rel_diff)
        max_rel_error_idx = np.unravel_index(
            np.argmax(rel_diff), rel_diff.shape)
        return max_rel_error, max_rel_error_idx

    def compute_mean_abs_error(a, b):
        abs_diff = np.abs(a - b)
        mean_abs_error = np.mean(abs_diff)
        return mean_abs_error

    def compute_mean_rel_error(a, b):
        denom = np.maximum(np.abs(a), np.abs(b))
        denom[denom == 0.0] = 1.0
        rel_diff = np.abs(a - b) / denom
        rel_diff[np.isnan(rel_diff)] = 0
        mean_rel_error = np.mean(rel_diff)
        return mean_rel_error

    if len(inputs) == 0:
        raise ValueError("No differentiable inputs available")
    if len(outputs) == 0:
        raise ValueError("No differentiable outputs available")
    jac_ad, ad_in, ad_out = kernel_jacobian(
        kernel, dim, inputs, outputs, max_outputs_per_var=max_outputs_per_var)
    jac_fd = kernel_jacobian_fd(
        kernel, dim, inputs, outputs, eps=eps, max_fd_dims_per_var=max_fd_dims_per_var)
    # assert_np_equal(jac_ad, jac_fd, tol=tol)
    result = np.allclose(jac_ad, jac_fd, atol=atol, rtol=rtol)
    max_abs_error, max_abs_error_idx = compute_max_abs_error(jac_ad, jac_fd)
    labels = find_variable_names(max_abs_error_idx)
    print(f"Max error: {max_abs_error} at {max_abs_error_idx} ({labels[0]} -> {labels[1]}): {jac_ad[max_abs_error_idx]} vs {jac_fd[max_abs_error_idx]}")
    max_rel_error, max_rel_error_idx = compute_max_rel_error(jac_ad, jac_fd)
    labels = find_variable_names(max_rel_error_idx)
    print(f"Max relative error: {max_rel_error} at {max_rel_error_idx} ({labels[0]} -> {labels[1]}): {jac_ad[max_rel_error_idx]} vs {jac_fd[max_rel_error_idx]}")

    # compute relative condition number
    # ||J(x)|| / (||f(x)|| / ||x||)
    nfx = np.linalg.norm(ad_out, ord=2)
    if nfx > 0:
        rel_condition_number = np.linalg.norm(jac_ad, ord='fro')*np.linalg.norm(ad_in, ord=2) / nfx
    else:
        rel_condition_number = np.linalg.norm(jac_ad, ord='fro')
    print(f"Relative condition number: {rel_condition_number}")

    # compute condition numbers
    cond_stat = {
        "total": np.linalg.cond(jac_ad),
        "individual": {}
    }
    max_abs_error_stat = {
        "total": max_abs_error,
        "individual": {}
    }
    max_rel_error_stat = {
        "total": max_rel_error,
        "individual": {}
    }
    mean_abs_error_stat = {
        "total": compute_mean_abs_error(jac_ad, jac_fd),
        "individual": {}
    }
    mean_rel_error_stat = {
        "total": compute_mean_rel_error(jac_ad, jac_fd),
        "individual": {}
    }

    if tabulate_errors:
        headers = ["Input", "Output", "Jac Block", "Sensitivity", "Max Rel Error", "Row", "Col", "AD", "FD"]
        table = [headers]
    highlight_xs = []
    highlight_ys = []
    for input_tick, input_label in zip(input_ticks, input_ticks_labels):
        for output_tick, output_label in zip(output_ticks, output_ticks_labels):
            input_len = min(input_lengths[input_label], max_fd_dims_per_var)
            output_len = min(output_lengths[output_label], max_outputs_per_var)
            jac_ad_sub = jac_ad[output_tick:output_tick+output_len, input_tick:input_tick+input_len]
            jac_fd_sub = jac_fd[output_tick:output_tick+output_len, input_tick:input_tick+input_len]
            cond_stat["individual"][(input_label, output_label)] = np.linalg.cond(jac_ad_sub)
            max_abs = compute_max_abs_error(jac_ad_sub, jac_fd_sub)
            max_rel = compute_max_rel_error(jac_ad_sub, jac_fd_sub)
            max_abs_error_stat["individual"][(input_label, output_label)] = max_abs[0]
            max_rel_error_stat["individual"][(input_label, output_label)] = max_rel[0]
            mean_abs_error_stat["individual"][(input_label, output_label)] = compute_mean_abs_error(jac_ad_sub, jac_fd_sub)
            mean_rel_error_stat["individual"][(input_label, output_label)] = compute_mean_rel_error(jac_ad_sub, jac_fd_sub)            
            actual_idx = (max_rel[1][0] + output_tick, max_rel[1][1] + input_tick)
            if max_rel[0] > 0.0:
                highlight_xs.append(actual_idx[1])  # swap because row is vertical
                highlight_ys.append(actual_idx[0])
            if tabulate_errors:
                # add the index offsets
                table.append([input_label, output_label,
                              f"[{output_tick}:{output_tick+output_len}, {input_tick}:{input_tick+input_len}]",
                              cond_stat["individual"][(input_label, output_label)],
                              max_rel[0],
                              actual_idx[0], actual_idx[1],
                              jac_ad[actual_idx], jac_fd[actual_idx]])


    stats = {
        "sensitivity": {
            "Jacobian Condition Number": cond_stat,
        },
        "accuracy": {
            "Max Absolute Jacobian Error": max_abs_error_stat,
            "Max Relative Jacobian Error": max_rel_error_stat,
            "Mean Absolute Jacobian Error": mean_abs_error_stat,
            "Mean Relative Jacobian Error": mean_rel_error_stat,
        },
        "jacobian": {
            "ad": jac_ad,
            "fd": jac_fd,
        }
    }

    if tabulate_errors:
        try:
            from tabulate import tabulate
            print(tabulate(table, headers="firstrow"))
        except ImportError:
            print("Install tabulate via `pip install tabulate` to print errors")

    if not result and plot_jac_on_fail:
        plot_jacobian_comparison(
            jac_ad, jac_fd,
            f"{kernel.key} Jacobian", 
            input_ticks, input_ticks_labels,
            output_ticks, output_ticks_labels,
            highlight_xs, highlight_ys)

    return result, stats


def make_struct_of_arrays(xs):
    """Convert a nested dict of list of arrays into a struct of arrays."""

    def insert(d, target):
        for k, v in d.items():
            if isinstance(v, dict):
                insert(v, target[k])
            else:
                target[k].append(v)

    total = defaultdict(list)
    for x in xs:
        insert(x, total)
    return total


def check_backward_pass(
    tape: wp.Tape,
    visualize_graph=True,
    check_jacobians=True,
    plot_jac_on_fail=False,
    plotting: Literal["matplotlib", "plotly", "none"] = "matplotlib",
    track_inputs=[],
    track_outputs=[],
    track_input_names=[],
    track_output_names=[],
    ignore_kernels=set(),
    ):
    """
    Runs various checks of the backward pass given the tape of recorded kernel launches.
    """

    def add_to_struct_of_arrays(d, target):
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in target:
                    target[k] = defaultdict(list)
                add_to_struct_of_arrays(v, target[k])
            else:
                target[k].append(v)

    import networkx as nx
    G = nx.DiGraph()
    node_labels = {}
    edge_labels = {}
    kernel_launch_count = defaultdict(int)
    # array -> list of kernels that modify it
    manipulated_nodes = defaultdict(list)
    kernel_nodes = set()
    array_nodes = set()

    input_output_ptr = set()
    for input in track_inputs:
        input_output_ptr.add(input.ptr)
    for output in track_outputs:
        input_output_ptr.add(output.ptr)

    def add_node(G, x, name):
        nonlocal node_labels
        if x.ptr in node_labels:
            if x.ptr not in input_output_ptr:
                # update name unless it is an input/output array
                node_labels[x.ptr] = name
            return
        nondifferentiable = (x.dtype in [
                             wp.int8, wp.uint8, wp.int16, wp.uint16, wp.int32, wp.uint32, wp.int64, wp.uint64])
        G.add_node(x.ptr, name=name, requires_grad=x.requires_grad,
                   is_kernel=False, nondifferentiable=nondifferentiable)
        node_labels[x.ptr] = name

    for i, x in enumerate(track_inputs):
        if i < len(track_input_names):
            name = track_input_names[i]
        else:
            name = f"input_{i}"
        add_node(G, x, name)
    for i, x in enumerate(track_outputs):
        if i < len(track_output_names):
            name = track_output_names[i]
        else:
            name = f"output_{i}"
        add_node(G, x, name)
    for launch in tape.launches:
        kernel, dim, inputs, outputs, device = tuple(launch)
        kernel_name = f"{kernel.key}:{kernel_launch_count[kernel.key]}"
        kernel_nodes.add(kernel_name)
        # store kernel as node with requires_grad so that the path search works
        G.add_node(kernel_name, is_kernel=True, dim=dim, requires_grad=True)
        node_labels[kernel_name] = kernel_name
        input_arrays = []
        for id, x in enumerate(inputs):
            name = kernel.adj.args[id].label
            if isinstance(x, wp.array):
                add_node(G, x, name)
                input_arrays.append(x.ptr)
            elif isinstance(x, wp.codegen.StructInstance):
                for varname in x._struct_.vars:
                    var = getattr(x, varname)
                    if isinstance(var, wp.array):
                        add_node(G, var, f"{name}.{varname}")
                        input_arrays.append(var.ptr)
        output_arrays = []
        for id, x in enumerate(outputs):
            name = kernel.adj.args[id + len(inputs)].label
            if isinstance(x, wp.array):
                add_node(G, x, name)
                output_arrays.append(x.ptr)
            elif isinstance(x, wp.codegen.StructInstance):
                for varname in x._struct_.vars:
                    var = getattr(x, varname)
                    if isinstance(var, wp.array):
                        add_node(G, var, f"{name}.{varname}")
                        output_arrays.append(var.ptr)
        for input_x in input_arrays:
            G.add_edge(input_x, kernel_name)
        for output_x in output_arrays:
            # track how many kernels modify each array
            manipulated_nodes[output_x].append(kernel.key)
            G.add_edge(kernel_name, output_x)

        kernel_launch_count[kernel.key] += 1

    for x in node_labels:
        if x not in kernel_nodes:
            array_nodes.add(x)

    for x in track_inputs:
        for y in track_outputs:
            try:
                paths = nx.all_shortest_paths(G, x.ptr, y.ptr)
                all_differentiable = True
                for path in paths:
                    # XXX all arrays up until the last one have to be differentiable
                    if not all([G.nodes[i]["requires_grad"] or G.nodes[i]["nondifferentiable"] for i in path[:-1]]):
                        print(FontColors.WARNING +
                            f"Warning: nondifferentiable node on path from {node_labels[x.ptr]} to {node_labels[y.ptr]} via [{' -> '.join([node_labels[p] for p in path])}]."
                            + FontColors.ENDC)
                        print(
                            f"Nondifferentiable array(s): [{', '.join([node_labels[p] for p in path if not G.nodes[p]['requires_grad']])}]")
                        all_differentiable = False
                if all_differentiable:
                    many_overwrites = set(node for node in path if len(
                        manipulated_nodes[node]) > 1)
                    if len(many_overwrites) > 0:
                        print(FontColors.WARNING +
                            f"Warning: multiple kernels manipulate array(s) on path from {node_labels[x.ptr]} to {node_labels[y.ptr]}." + FontColors.ENDC)
                        for node in many_overwrites:
                            print(
                                f"\tArray {node_labels[node]} is manipulated by kernels [{', '.join([kernel for kernel in manipulated_nodes[node]])}].")
                    else:
                        print(FontColors.OKGREEN +
                            f"Path from {node_labels[x.ptr]} to {node_labels[y.ptr]} is differentiable." + FontColors.ENDC)
            except nx.NetworkXNoPath:
                print(FontColors.FAIL +
                    f"Error: there is no computation path from {node_labels[x.ptr]} to {node_labels[y.ptr]}" + FontColors.ENDC)

    if visualize_graph:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        try:
            pos = nx.nx_agraph.graphviz_layout(
                G, prog='neato', args='-Grankdir="LR" -Gnodesep="5" -Granksep="10"')
            # pos = nx.spring_layout(G, seed=42, pos=pos, iterations=1)
        except:
            print(
                "Warning: could not use graphviz to layout graph. Falling back to spring layout.")
            print("To get better layouts, install graphviz and pygraphviz.")
            pos = nx.spring_layout(G)

        fig = plt.figure()
        fig.canvas.manager.set_window_title("Kernel launch graph")
        array_nodes = list(array_nodes)
        kernel_nodes = list(kernel_nodes)
        node_colors = []
        for x in array_nodes:
            if len(manipulated_nodes[x]) > 1:
                node_colors.append("salmon")
            elif G.nodes[x]["requires_grad"]:
                node_colors.append("lightskyblue")
            else:
                node_colors.append("lightgray")

        handles = [
            mpl.patches.Patch(color="salmon", label="multiple overwrites"),
            mpl.patches.Patch(color="lightskyblue", label="requires grad"),
            mpl.patches.Patch(color="lightgray", label="no grad"),
            mpl.patches.Patch(color="yellow", label="kernel"),
        ]
        plt.legend(handles=handles)

        default_draw_args = dict(
            alpha=0.9, edgecolors="black", linewidths=0.5, node_size=1000)
        # first draw kernels
        nx.draw_networkx_nodes(G, pos, nodelist=kernel_nodes, node_color='yellow', node_shape='s', **default_draw_args)
        # then draw arrays
        nx.draw_networkx_nodes(G, pos, nodelist=array_nodes, node_color=node_colors, **default_draw_args)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True, edge_color='black', node_size=1000)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={
                uv: f"{d['kernel']}:{d['launch']}" for uv, d in edge_labels.items()},
            font_color='darkslategray'
        )
        plt.axis('off')
        plt.show()

    stats = {}
    kernel_names = set()
    manipulated_vars = {}
    problematic_vars = set()
    chart_code = []
    chart_vars = {}
    kernel_launch_count = defaultdict(int)
    hide_non_arrays = True
    for launch in tape.launches:
        kernel, dim, inputs, outputs, device = tuple(launch)
        if check_jacobians and kernel.key not in ignore_kernels:
            msg = f"Checking Jacobian of kernel \"{kernel.key}\" (launch {kernel_launch_count[kernel.key]})..."
            print("".join(["#"] * len(msg)))
            print(FontColors.OKCYAN + msg + FontColors.ENDC)
            try:
                result, kernel_stats = check_kernel_jacobian(
                    kernel, dim, inputs, outputs, plot_jac_on_fail=plot_jac_on_fail, atol=1.0)
                print(result)
                if kernel.key not in stats:
                    stats[kernel.key] = defaultdict(list)
                add_to_struct_of_arrays(kernel_stats, stats[kernel.key])
            except Exception as e:
                print(FontColors.FAIL + f"Error while checking jacobian of kernel {kernel.key}: {e}" + FontColors.ENDC)

        kernel_names.add(kernel.key)

        def sanitize_name(s):
            # XXX there seems to be a bug in mermaid where it tries to interpret
            # a node name as a link target if it contains the word "parent"
            return s.replace('parent', 'pärent')

        kernel_id = f"{kernel.key}{kernel_launch_count[kernel.key]}"
        chart_code.append(
            f"{kernel_id}[[{sanitize_name(kernel.key)}]]:::kernel;")
        kernel_launch_count[kernel.key] += 1

        input_nodes = []
        for id, x in enumerate(inputs):
            name = sanitize_name(kernel.adj.args[id].label)
            if isinstance(x, wp.array):
                if x.requires_grad:
                    input_nodes.append(f"a{x.ptr}")
                    if x.ptr not in manipulated_vars:
                        chart_vars[x.ptr] = f"a{x.ptr}([{name}]):::grad;"
                else:
                    input_nodes.append(f"a{x.ptr}")
                    chart_vars[x.ptr] = f"a{x.ptr}([{name}]):::nograd;"
            elif isinstance(x, wp.codegen.StructInstance):
                for varname in x._struct_.vars:
                    var = getattr(x, varname)
                    if isinstance(var, wp.array):
                        if var.requires_grad:
                            input_nodes.append(f"a{var.ptr}")
                            if var.ptr not in manipulated_vars:
                                chart_vars[var.ptr] = f"a{var.ptr}([{name}.{varname}]):::grad;"
                        else:
                            input_nodes.append(f"a{var.ptr}")
                            chart_vars[var.ptr] = f"a{var.ptr}([{name}.{varname}]):::nograd;"
                    elif not hide_non_arrays:
                        input_nodes.append(f"a{name}.{varname}([{name}.{varname}])")
            elif not hide_non_arrays:
                input_nodes.append(f"a{name}([{name}])")
        output_nodes = []
        for id, x in enumerate(outputs):
            name = sanitize_name(kernel.adj.args[id + len(inputs)].label)
            if isinstance(x, wp.array):
                if x.requires_grad:
                    output_nodes.append(f"a{x.ptr}")
                    if x.ptr in manipulated_vars:
                        chart_vars[x.ptr] = f"a{x.ptr}([{name}]):::problem;"
                        # print(
                        #     f"WARNING: variable {name} requires grad and is manipulated by kernels {kernel.key} and {manipulated_vars[x.ptr]}.")
                        problematic_vars.add(f"a{x.ptr}")
                    else:
                        chart_vars[x.ptr] = f"a{x.ptr}([{name}]):::grad;"
                        manipulated_vars[x.ptr] = kernel.key
                else:
                    output_nodes.append(f"a{x.ptr}")
                    chart_vars[x.ptr] = f"a{x.ptr}([{name}]):::nograd;"
            elif not hide_non_arrays:
                output_nodes.append(f"a{name}([{name}])")

        chart_code.append(f"subgraph graph{kernel_id}[{kernel.key}]")
        # chart_code.append(f"{' & '.join(input_nodes)} --> {kernel_id};")
        # chart_code.append(f"{kernel_id} --> {' & '.join(output_nodes)};")
        for node in input_nodes:
            chart_code.append(f"{node} --> {kernel_id};")
        for node in output_nodes:
            if node in problematic_vars:
                chart_code.append(f"{kernel_id} -.-> {node};")
            else:
                chart_code.append(f"{kernel_id} --> {node};")
        chart_code.append("end")

    chart_code.append(
        "classDef kernel fill:#222,color:#fff,stroke:#333,stroke-width:4px")
    chart_code.append("classDef grad fill:#efe,stroke:#060")
    chart_code.append("classDef nograd fill:#ffe,stroke:#630")
    chart_code.append("classDef problem fill:#f65,color:#900,stroke:#900")
    chart_code.append("class a fill:#fff,stroke:#000")

    chart = "%%{init: {'flowchart': { 'curve': 'curve' }, 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffeedd'}}}%%\n"
    chart += "flowchart LR;\n"
    chart += "\n".join([f"\t{line}" for line in chart_vars.values()])
    chart += "\n"
    chart += "\n".join([f"\t{line}" for line in chart_code])

    # print(chart)

    # import dash
    # from dash_extensions import Mermaid
    # app = dash.Dash()
    # app.layout = Mermaid(chart=chart)

    # app.run_server()

    if check_jacobians:
        # plot evolution of Jacobian statistics
        if plotting == "matplotlib":
            from itertools import chain
            any_stat = next(iter(stats.values()))
            all_stats_names = chain.from_iterable([any_stat[cat].keys() for cat in ["sensitivity", "accuracy"]])
            all_stats_names = [key for key in all_stats_names if key not in ("total", "individual")]
            for kernel_name, stat in stats.items():
                import matplotlib.pyplot as plt
                num = len(all_stats_names)
                ncols = int(np.ceil(np.sqrt(num)))
                nrows = int(np.ceil(num / float(ncols)))
                fig, axes = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    figsize=(ncols * 5.5, nrows * 3.5),
                    squeeze=False,
                )
                fig.canvas.set_window_title(kernel_name)
                plt.suptitle(kernel_name, fontsize=16, fontweight="bold")
                for dim in range(ncols * nrows):
                    ax = axes[dim // ncols, dim % ncols]
                    if dim >= num:
                        ax.axis("off")
                        continue
                kernel_stats = list(chain.from_iterable([stat[cat].items() for cat in ["sensitivity", "accuracy"]]))
                for dim, (stat_name, cond) in enumerate(kernel_stats):
                    ax = axes[dim // ncols, dim % ncols]
                    ax.set_title(f"{stat_name}")
                    marker = "o" if len(cond["total"]) < 10 else None
                    ax.plot(cond["total"], label="total", c="k", zorder=2, marker=marker)
                    ax.set_yscale("log")
                    for key, value in cond["individual"].items():
                        marker = "o" if len(value) < 10 else None
                        ax.plot(value, label=f"{key[0]} $\\to$ {key[1]}", zorder=1, marker=marker)
                    if dim == len(kernel_stats)-1:
                        ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), fancybox=True, shadow=True, ncol=2)
                    ax.grid()
                plt.subplots_adjust(hspace=0.2, wspace=0.2,
                                    top=0.9, left=0.1, right=0.9, bottom=0.1)
                plt.show()
        elif plotting == "plotly":
            import dash
            from dash import Dash, dcc, html
            from dash.dependencies import Input, Output, State
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px

            external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
            app = Dash(__name__, external_stylesheets=external_stylesheets)

            kernel_names = sorted(list(kernel_names))

            colors = px.colors.qualitative.Dark24

            app.layout = html.Div([
                dcc.Store(id='appstate', data={
                        "kernel": kernel_names[0], "mode": "jacobian"}),
                html.Div([
                    dcc.Dropdown(
                        options=[{"label": name, "value": name}
                                for name in kernel_names],
                        value=kernel_names[0],
                        placeholder="Select a kernel",
                        id="kernel-dropdown",
                    ),
                ], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Tabs(id="mode-selector", value="jacobian", children=[
                        dcc.Tab(label="Jacobian", value="jacobian"),
                        dcc.Tab(label="Sensitivity", value="sensitivity"),
                        dcc.Tab(label="Accuracy", value="accuracy"),
                    ])
                ], style={'width': '49%', 'display': 'inline-block'}),
                html.Div(id='view-content')
            ])

            @app.callback(Output('appstate', 'data'), Input('kernel-dropdown', 'value'), Input('mode-selector', 'value'), State('appstate', 'data'))
            def update_kernel(kernel, mode, data):
                if kernel is not None:
                    data["kernel"] = kernel
                if mode is not None:
                    data["mode"] = mode
                return data

            @app.callback(Output('view-content', 'children'), Input('appstate', 'data'), State('appstate', 'data'))
            def update_view(arg, data):
                print("selection:", data)
                selected_kernel = data["kernel"]
                selected_mode = data["mode"]
                selected_stats = stats[selected_kernel][selected_mode]
                if selected_mode == "jacobian":
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=["AD Jacobian",
                                        "FD Jacobian", "Difference"],
                    )
                    fig.add_trace(
                        px.imshow(selected_stats["ad"][0]).data[0], row=1, col=1)
                    fig.add_trace(
                        px.imshow(selected_stats["fd"][0]).data[0], row=1, col=2)
                    fig.add_trace(px.imshow(
                        selected_stats["ad"][0]-selected_stats["fd"][0]).data[0], row=1, col=3)
                    fig.update_layout(coloraxis=dict(colorscale='RdBu_r'))
                else:
                    selected_stat_items = selected_stats.items()
                    num = len(selected_stat_items)
                    ncols = int(np.ceil(np.sqrt(num)))
                    nrows = int(np.ceil(num / float(ncols)))
                    fig = make_subplots(
                        rows=nrows, cols=ncols,
                        subplot_titles=list(selected_stats.keys()),
                        shared_xaxes=True,
                        vertical_spacing=0.07,
                        horizontal_spacing=0.05)

                    previous_labels = set()  # avoid duplicate legend entries
                    for dim, (stat_name, stat) in enumerate(selected_stat_items):
                        row = dim // ncols + 1
                        col = dim % ncols + 1
                        fig.add_trace(go.Scatter(
                            y=stat["total"], name="total", line=dict(color="#000000"), legendgroup='group0', showlegend=(dim == 0)), row=row, col=col)
                        fig.update_yaxes(type="log", row=row, col=col)
                        for i, (key, value) in enumerate(stat["individual"].items()):
                            label = "{0} → {1}".format(key[0], key[1])
                            fig.add_trace(go.Scatter(
                                x=np.arange(len(value)),
                                y=value,
                                name=label,
                                legendgroup=f'group{label}',
                                showlegend=(label not in previous_labels),
                                hoverlabel=dict(namelength=-1),
                                line=dict(color=colors[i % len(colors)])), row=row, col=col)
                            previous_labels.add(label)

                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                )

                return dcc.Graph(figure=fig, style={'width': '100vw', 'height': '90vh'})

            # @app.callback(Output('view-content', 'children'), Input('tabs', 'value'))
            # def stat_selection(tab):
            #     try:
            #         stat_id = int(tab)
            #     except:
            #         stat_id = 0
            #     stat_name, stat = stat_items[stat_id]
            #     num = len(stat)
            #     ncols = int(np.ceil(np.sqrt(num)))
            #     nrows = int(np.ceil(num / float(ncols)))
            #     fig = make_subplots(
            #         rows=nrows, cols=ncols,
            #         subplot_titles=list(stat.keys()))

            #     previous_labels = set()  # avoid duplicate legend entries
            #     for dim, (kernel_key, cond) in enumerate(stat.items()):
            #         row = dim // ncols + 1
            #         col = dim % ncols + 1
            #         fig.add_trace(go.Scatter(
            #             y=cond["total"], name="total", line=dict(color="#000000"), legendgroup='group0', showlegend=(dim==0)), row=row, col=col)
            #         fig.update_yaxes(type="log", row=row, col=col)
            #         for key, value in cond["individual"].items():
            #             label = "{0} → {1}".format(key[0], key[1])
            #             fig.add_trace(go.Scatter(
            #                 x=np.arange(len(value)),
            #                 y=value,
            #                 name=label,
            #                 legendgroup=f'group{label}',
            #                 showlegend=(label not in previous_labels),
            #                 hoverlabel=dict(namelength=-1)), row=row, col=col)
            #             previous_labels.add(label)

            #     return dcc.Graph(figure=fig, style={'width': '100vw', 'height': '90vh'})

            app.title = "Warp backward pass statistics"
            app.run()
