# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *
from warp.tests.grad_utils import check_kernel_jacobian

wp.init()

@wp.kernel
def scalar_grad(x: wp.array(dtype=float),
                y: wp.array(dtype=float)):

    y[0] = x[0]**2.0



def test_scalar_grad(test, device):

    x = wp.array([3.0], dtype=float, device=device, requires_grad=True)
    y = wp.zeros_like(x)

    tape = wp.Tape()
    with tape:
        wp.launch(scalar_grad, dim=1, inputs=[x, y], device=device)

    tape.backward(y)

    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))
   



@wp.kernel
def for_loop_grad(n: int, 
                  x: wp.array(dtype=float),
                  s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):
        sum = sum + x[i]*2.0

    s[0] = sum


def test_for_loop_grad(test, device):

    n = 32
    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad, dim=1, inputs=[n, x, sum], device=device)
   
    # ensure forward pass outputs correct
    assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))

    tape.backward(loss=sum)
    
    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(tape.gradients[x].numpy(), 2.0*val)


def test_for_loop_graph_grad(test, device):

    n = 32
    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    wp.capture_begin()

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad, dim=1, inputs=[n, x, sum], device=device)
   
    tape.backward(loss=sum)

    graph = wp.capture_end()

    wp.capture_launch(graph)
    wp.synchronize()
    
    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(x.grad.numpy(), 2.0*val)

    wp.capture_launch(graph)
    wp.synchronize()    

@wp.kernel
def for_loop_nested_if_grad(n: int, 
                            x: wp.array(dtype=float),
                            s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):

        if i < 16:
            if i < 8:
                sum = sum + x[i]*2.0
            else:
                sum = sum + x[i]*4.0
        else:
            if i < 24:
                sum = sum + x[i]*6.0
            else:
                sum = sum + x[i]*8.0


    s[0] = sum


def test_for_loop_nested_if_grad(test, device):

    n = 32
    val = np.ones(n, dtype=np.float32)

    expected_val = [2., 2., 2., 2., 2., 2., 2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 6., 6., 6., 6., 6., 6., 6., 6., 8., 8., 8., 8., 8., 8., 8., 8.]
    expected_grad = [2., 2., 2., 2., 2., 2., 2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 6., 6., 6., 6., 6., 6., 6., 6., 8., 8., 8., 8., 8., 8., 8., 8.]

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_nested_if_grad, dim=1, inputs=[n, x, sum], device=device)
   
    assert_np_equal(sum.numpy(), np.sum(expected_val))

    tape.backward(loss=sum)
    
    assert_np_equal(sum.numpy(), np.sum(expected_val))
    assert_np_equal(tape.gradients[x].numpy(), np.array(expected_grad))



@wp.kernel
def for_loop_grad_nested(n: int, 
                         x: wp.array(dtype=float),
                         s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):
        for j in range(n):
            sum = sum + x[i*n + j]*float(i*n + j) + 1.0

    s[0] = sum


def test_for_loop_nested_for_grad(test, device):
    
    x = wp.zeros(9, dtype=float, device=device, requires_grad=True)
    s = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad_nested, dim=1, inputs=[3, x, s], device=device)

    tape.backward(s)

    assert_np_equal(s.numpy(), np.array([9.0]))
    assert_np_equal(tape.gradients[x].numpy(), np.arange(0.0, 9.0, 1.0))


# differentiating thought most while loops is not supported
# since doing things like i = i + 1 breaks adjointing

# @wp.kernel
# def while_loop_grad(n: int, 
#                     x: wp.array(dtype=float),
#                     c: wp.array(dtype=int),
#                     s: wp.array(dtype=float)):

#     tid = wp.tid()

#     while i < n:
#         s[0] = s[0] + x[i]*2.0
#         i = i + 1

        

# def test_while_loop_grad(test, device):

#     n = 32
#     x = wp.array(np.ones(n, dtype=np.float32), device=device, requires_grad=True)
#     c = wp.zeros(1, dtype=int, device=device)
#     sum = wp.zeros(1, dtype=wp.float32, device=device)

#     tape = wp.Tape()
#     with tape:
#         wp.launch(while_loop_grad, dim=1, inputs=[n, x, c, sum], device=device)
   
#     tape.backward(loss=sum)

#     assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))
#     assert_np_equal(tape.gradients[x].numpy(), 2.0*np.ones_like(x.numpy()))



@wp.kernel
def preserve_outputs(n: int, 
                     x: wp.array(dtype=float),
                     c: wp.array(dtype=float),
                     s1: wp.array(dtype=float),
                     s2: wp.array(dtype=float)):

    tid = wp.tid()

    # plain store
    c[tid] = x[tid]*2.0

    # atomic stores
    wp.atomic_add(s1, 0, x[tid]*3.0)
    wp.atomic_sub(s2, 0, x[tid]*2.0)


# tests that outputs from the forward pass are
# preserved by the backward pass, i.e.: stores
# are omitted during the forward reply
def test_preserve_outputs_grad(test, device):

    n = 32

    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    c = wp.zeros_like(x)
    
    s1 = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    s2 = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(preserve_outputs, dim=n, inputs=[n, x, c, s1, s2], device=device)

    # ensure forward pass results are correct
    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val*2.0)
    assert_np_equal(s1.numpy(), np.array(3.0*n))
    assert_np_equal(s2.numpy(), np.array(-2.0*n))
    
    # run backward on first loss
    tape.backward(loss=s1)

    # ensure inputs, copy and sum are unchanged by backwards pass
    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val*2.0)
    assert_np_equal(s1.numpy(), np.array(3.0*n))
    assert_np_equal(s2.numpy(), np.array(-2.0*n))

    # ensure gradients are correct
    assert_np_equal(tape.gradients[x].numpy(), 3.0*val)

    # run backward on second loss
    tape.zero()
    tape.backward(loss=s2)

    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val*2.0)
    assert_np_equal(s1.numpy(), np.array(3.0*n))
    assert_np_equal(s2.numpy(), np.array(-2.0*n))

    # ensure gradients are correct
    assert_np_equal(tape.gradients[x].numpy(), -2.0*val)

def gradcheck(func, func_name, inputs, device, outputs=None, eps=1e-4, tol=1e-2):
    """
    Checks that the gradient of the Warp kernel is correct by comparing it to the
    numerical gradient computed using finite differences.
    """
    
    module = wp.get_module(func.__module__)
    kernel = wp.Kernel(func=func, key=func_name, module=module)

    if outputs is None:
        outputs = [wp.zeros(1, dtype=wp.float32, device=device)]
    print(f"Checking gradient of {func_name} on {device}...")
    result, stats = check_kernel_jacobian(
        kernel, dim=1, inputs=inputs, outputs=outputs, eps=eps, atol=tol,
        plot_jac_on_fail=False, tabulate_errors=True, warn_about_missing_requires_grad=False)
    assert result

def test_vector_math_grad(test, device):
    np.random.seed(123)
    
    # test unary operations
    for dim, vec_type in [(2, wp.vec2), (3, wp.vec3), (4, wp.vec4), (4, wp.quat)]:
        def check_length(vs: wp.array(dtype=vec_type), out: wp.array(dtype=float)):
            out[0] = wp.length(vs[0])

        def check_length_sq(vs: wp.array(dtype=vec_type), out: wp.array(dtype=float)):
            out[0] = wp.length_sq(vs[0])

        def check_normalize(vs: wp.array(dtype=vec_type), out: wp.array(dtype=float)):
            out[0] = wp.length_sq(wp.normalize(vs[0]))  # compress to scalar output

        # run the tests with 5 different random inputs
        for _ in range(5):
            x = wp.array(np.random.randn(1, dim).astype(np.float32), dtype=vec_type, device=device)
            gradcheck(check_length, f"check_length_{vec_type.__name__}", [x], device)
            gradcheck(check_length_sq, f"check_length_sq_{vec_type.__name__}", [x], device)
            gradcheck(check_normalize, f"check_normalize_{vec_type.__name__}", [x], device)

def test_matrix_math_grad(test, device):
    np.random.seed(123)
    
    # test unary operations
    for dim, mat_type in [(2, wp.mat22), (3, wp.mat33), (4, wp.mat44)]:
        def check_determinant(vs: wp.array(dtype=mat_type), out: wp.array(dtype=float)):
            out[0] = wp.determinant(vs[0])

        def check_trace(vs: wp.array(dtype=mat_type), out: wp.array(dtype=float)):
            out[0] = wp.trace(vs[0])

        # run the tests with 5 different random inputs
        for _ in range(5):
            x = wp.array(np.random.randn(1, dim, dim).astype(np.float32), ndim=1, dtype=mat_type, device=device)
            gradcheck(check_determinant, f"check_determinant_{mat_type.__name__}", [x], device)
            gradcheck(check_trace, f"check_trace_{mat_type.__name__}", [x], device)

def test_3d_math_grad(test, device):
    np.random.seed(123)
    
    # test binary operations
    def check_cross(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
        out[0] = wp.cross(vs[0], vs[1])

    def check_dot(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        out[0] = wp.dot(vs[0], vs[1])

    def check_mat33(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        a = vs[0]
        b = vs[1]
        c = wp.cross(a, b)
        m = wp.mat33(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2])
        out[0] = wp.determinant(m)

    def check_trace_diagonal(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        a = vs[0]
        b = vs[1]
        c = wp.cross(a, b)
        m = wp.mat33(
            1.0 / (a[0] + 10.0), 0.0, 0.0,
            0.0, 1.0 / (b[1] + 10.0), 0.0,
            0.0, 0.0, 1.0 / (c[2] + 10.0),
        )
        out[0] = wp.trace(m)

    def check_rot_rpy(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        v = vs[0]
        q = wp.quat_rpy(v[0], v[1], v[2])
        out[0] = wp.length(wp.quat_rotate(q, vs[1]))

    def check_rot_axis_angle(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        v = wp.normalize(vs[0])
        q = wp.quat_from_axis_angle(v, 0.5)
        out[0] = wp.length(wp.quat_rotate(q, vs[1]))

    def check_rot_quat_inv(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        v = vs[0]
        q = wp.normalize(wp.quat(v[0], v[1], v[2], 1.0))
        out[0] = wp.length(wp.quat_rotate_inv(q, vs[1]))

    # run the tests with 5 different random inputs
    for _ in range(5):
        x = wp.array(np.random.randn(2, 3).astype(np.float32), dtype=wp.vec3, device=device)
        gradcheck(check_cross, f"check_cross_3d", [x], device, outputs=[wp.zeros(1, dtype=wp.vec3, device=device)])
        gradcheck(check_dot, f"check_dot_3d", [x], device)
        gradcheck(check_mat33, f"check_mat33_3d", [x], device)
        gradcheck(check_trace_diagonal, f"check_trace_diagonal_3d", [x], device)
        gradcheck(check_rot_rpy, f"check_rot_rpy_3d", [x], device)
        gradcheck(check_rot_axis_angle, f"check_rot_axis_angle_3d", [x], device)
        gradcheck(check_rot_quat_inv, f"check_rot_quat_inv_3d", [x], device)

@wp.kernel
def spurious_assignment(xs: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    x = xs[0]
    y = x  # <--  y is not used anywhere, yet it causes the gradient of x to be zero!
    output[0] = x

@wp.kernel
def no_spurious_assignment(xs: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    x = xs[0]
    output[0] = x

def register(parent):

    devices = wp.get_devices()

    class TestGrad(parent):
        pass

    # add_function_test(TestGrad, "test_while_loop_grad", test_while_loop_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_nested_for_grad", test_for_loop_nested_for_grad, devices=devices)
    add_function_test(TestGrad, "test_scalar_grad", test_scalar_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_grad", test_for_loop_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_graph_grad", test_for_loop_graph_grad, devices=wp.get_cuda_devices())
    add_function_test(TestGrad, "test_for_loop_nested_if_grad", test_for_loop_nested_if_grad, devices=devices)
    add_function_test(TestGrad, "test_preserve_outputs_grad", test_preserve_outputs_grad, devices=devices)
    add_function_test(TestGrad, "test_vector_math_grad", test_vector_math_grad, devices=devices)
    add_function_test(TestGrad, "test_matrix_math_grad", test_matrix_math_grad, devices=devices)
    add_function_test(TestGrad, "test_3d_math_grad", test_3d_math_grad, devices=devices)

    # from grad_utils import check_kernel_jacobian
    # check_kernel_jacobian(
    #     spurious_assignment, 1,
    #     inputs=[wp.array([1, 2, 3], dtype=wp.vec3, device=wp.get_device())],
    #     outputs=[wp.array([0, 0, 0], dtype=wp.vec3, device=wp.get_device())],
    #     plot_jac_on_fail=True, atol=0.1)
    # check_kernel_jacobian(
    #     no_spurious_assignment, 1,
    #     inputs=[wp.array([1, 2, 3], dtype=wp.vec3, device=wp.get_device())],
    #     outputs=[wp.array([0, 0, 0], dtype=wp.vec3, device=wp.get_device())],
    #     plot_jac_on_fail=True, atol=-0.1)


    return TestGrad

def run():

    test_suite = unittest.TestSuite()
    result = unittest.TestResult()

    tests = [register(unittest.TestCase)]

    for test in tests:
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(test))

    # force rebuild of all kernels
    wp.build.clear_kernel_cache()

    # load all modules
    wp.force_load()

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    ret = not runner.run(test_suite).wasSuccessful()
    return ret


if __name__ == '__main__':
    ret = run()

    import sys
    sys.exit(ret)

