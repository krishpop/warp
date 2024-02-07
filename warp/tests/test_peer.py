# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


def get_device_pair_with_peer_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if wp.is_peer_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def get_device_pair_without_peer_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if not wp.is_peer_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def test_peer_access_self(test, device):

    device = wp.get_device(device)

    assert device.is_cuda

    # device can access self
    can_access = wp.is_peer_access_supported(device, device)
    test.assertTrue(can_access)

    # setting peer access to self is a no-op
    wp.set_peer_access_enabled(device, device, True)
    wp.set_peer_access_enabled(device, device, False)

    # should always be enabled
    enabled = wp.is_peer_access_enabled(device, device)        
    test.assertTrue(enabled)


def test_peer_access(test, _):

    peer_pair = get_device_pair_with_peer_access_support()

    if peer_pair:
        target_device, peer_device = peer_pair

        was_enabled = wp.is_peer_access_enabled(target_device, peer_device)

        if was_enabled:
            # try disabling
            wp.set_peer_access_enabled(target_device, peer_device, False)
            is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
            test.assertFalse(is_enabled)

            # try re-enabling
            wp.set_peer_access_enabled(target_device, peer_device, True)
            is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
            test.assertTrue(is_enabled)
        else:
            # try enabling
            wp.set_peer_access_enabled(target_device, peer_device, True)
            is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
            test.assertTrue(is_enabled)

            # try re-disabling
            wp.set_peer_access_enabled(target_device, peer_device, False)
            is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
            test.assertFalse(is_enabled)


def test_peer_access_exceptions(test, _):

    # get a CUDA device pair without peer access support
    peer_pair = get_device_pair_without_peer_access_support()

    if peer_pair:
        target_device, peer_device = peer_pair

        # querying is ok, but must return False
        test.assertFalse(wp.is_peer_access_enabled(target_device, peer_device))
    
        # enabling should raise RuntimeError
        with test.assertRaises(RuntimeError):
            wp.set_peer_access_enabled(target_device, peer_device, True)

        # disabling should not raise an error
        wp.set_peer_access_enabled(target_device, peer_device, False)

    # test CPU/CUDA errors
    if wp.is_cpu_available() and wp.is_cuda_available():

        # querying is ok, but must return False
        test.assertFalse(wp.is_peer_access_enabled("cuda:0", "cpu"))
        test.assertFalse(wp.is_peer_access_enabled("cpu", "cuda:0"))

        # enabling should raise ValueError
        with test.assertRaises(ValueError):
            wp.set_peer_access_enabled("cpu", "cuda:0", True)
        with test.assertRaises(ValueError):
            wp.set_peer_access_enabled("cuda:0", "cpu", True)

        # disabling should not raise an error
        wp.set_peer_access_enabled("cpu", "cuda:0", False)
        wp.set_peer_access_enabled("cuda:0", "cpu", False)


class TestPeer(unittest.TestCase):
    pass


cuda_test_devices = get_cuda_test_devices()

add_function_test(TestPeer, "test_peer_access_self", test_peer_access_self, devices=cuda_test_devices)

# MGPU tests
if get_device_pair_with_peer_access_support():
    add_function_test(TestPeer, "test_peer_access", test_peer_access)

# test access failures
add_function_test(TestPeer, "test_peer_access_exceptions", test_peer_access_exceptions)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
