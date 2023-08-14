# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from typing import Any


class SGD:
    """An implementation of the Stochastic Gradient Descent Optimizer
    It is designed to mimic Pytorch's version.
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """

    def __init__(self, params=None, lr=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        self.b = []  # momentum buffer
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.t = 0
        self.set_params(params)

    def set_params(self, params):
        self.params = params
        self.ndim_kernels = []
        if params is not None and type(params) == list and len(params) > 0:
            if len(self.b) != len(params):
                self.b = [None] * len(params)
            for i in range(len(params)):
                param = params[i]
                if self.b[i] is None or self.b[i].shape != param.shape or self.b[i].dtype != param.dtype:
                    self.b[i] = wp.zeros_like(param)
                ndim = param.ndim
                dtype = param.dtype

                if ndim == 1:
                    @wp.kernel(enable_backward=False)
                    def sgd_step_kernel(
                        g: wp.array(dtype=dtype, ndim=ndim),
                        b: wp.array(dtype=dtype, ndim=ndim),
                        lr: float,
                        weight_decay: float,
                        momentum: float,
                        damping: float,
                        nesterov: int,
                        t: int,
                        parameters: wp.array(dtype=dtype, ndim=ndim),
                    ):
                        i = wp.tid()
                        gt = g[i]
                        if weight_decay != 0.0:
                            gt += weight_decay * parameters[i]
                        if momentum != 0.0:
                            bt = b[i]
                            if t > 0:
                                bt = momentum * bt + (1.0 - damping) * gt
                            else:
                                bt = gt
                            if nesterov == 1:
                                gt += momentum * bt
                            else:
                                gt = bt
                            b[i] = bt
                        parameters[i] = parameters[i] - lr * gt
                if ndim == 2:
                    @wp.kernel(enable_backward=False)
                    def sgd_step_kernel(
                        g: wp.array(dtype=dtype, ndim=ndim),
                        b: wp.array(dtype=dtype, ndim=ndim),
                        lr: float,
                        weight_decay: float,
                        momentum: float,
                        damping: float,
                        nesterov: int,
                        t: int,
                        parameters: wp.array(dtype=dtype, ndim=ndim),
                    ):
                        i, j = wp.tid()
                        gt = g[i]
                        if weight_decay != 0.0:
                            gt[j] = gt[j] + weight_decay * parameters[i, j]
                        if momentum != 0.0:
                            bt = b[i]
                            if t > 0:
                                bt[j] = momentum * bt[j] + (1.0 - damping) * gt[j]
                            else:
                                bt = gt
                            if nesterov == 1:
                                gt[j] = gt[j] + momentum * bt[j]
                            else:
                                gt = bt
                            b[i] = bt
                        parameters[i, j] = parameters[i, j] - lr * gt[j]
                elif ndim == 3:
                    @wp.kernel(enable_backward=False)
                    def sgd_step_kernel(
                        g: wp.array(dtype=dtype, ndim=ndim),
                        b: wp.array(dtype=dtype, ndim=ndim),
                        lr: float,
                        weight_decay: float,
                        momentum: float,
                        damping: float,
                        nesterov: int,
                        t: int,
                        parameters: wp.array(dtype=dtype, ndim=ndim),
                    ):
                        i, j, k = wp.tid()
                        gt = g[i, j]
                        if weight_decay != 0.0:
                            gt[k] = gt[k] + weight_decay * parameters[i, j, k]
                        if momentum != 0.0:
                            bt = b[i, j]
                            if t > 0:
                                bt[k] = momentum * bt[k] + (1.0 - damping) * gt[k]
                            else:
                                bt = gt
                            if nesterov == 1:
                                gt[k] = gt[k] + momentum * bt[k]
                            else:
                                gt = bt
                            for k2 in range(gt.shape[2]):
                                b[i, j, k2] = bt[k2]
                        parameters[i, j, k] = parameters[i, j, k] - lr * gt[k]
                elif ndim == 4:
                    @wp.kernel(enable_backward=False)
                    def sgd_step_kernel(
                        g: wp.array(dtype=dtype, ndim=ndim),
                        b: wp.array(dtype=dtype, ndim=ndim),
                        lr: float,
                        weight_decay: float,
                        momentum: float,
                        damping: float,
                        nesterov: int,
                        t: int,
                        parameters: wp.array(dtype=dtype, ndim=ndim),
                    ):
                        i, j, k, l = wp.tid()
                        gt = g[i, j, k]
                        if weight_decay != 0.0:
                            gt[l] = gt[l] + weight_decay * parameters[i, j, k, l]
                        if momentum != 0.0:
                            bt = b[i, j, k]
                            if t > 0:
                                bt[l] = momentum * bt[l] + (1.0 - damping) * gt[l]
                            else:
                                bt = gt
                            if nesterov == 1:
                                gt[l] = gt[l] + momentum * bt[l]
                            else:
                                gt = bt
                            b[i, j, k] = bt
                        parameters[i, j, k, l] = parameters[i, j, k, l] - lr * gt[l]
                    
                self.ndim_kernels.append(sgd_step_kernel)

    def reset_internal_state(self):
        for b_i in self.b:
            b_i.zero_()
        self.t = 0

    def step(self, grad):
        assert self.params is not None
        for i in range(len(self.params)):
            self.step_detail(
                i, grad[i], self.b[i], self.lr, self.momentum, self.dampening, self.weight_decay, self.nesterov, self.t, self.params[i]
            )
        self.t = self.t + 1

    def step_detail(self, i, g, b, lr, momentum, dampening, weight_decay, nesterov, t, params):
        assert params.dtype == g.dtype
        assert params.dtype == b.dtype
        assert params.shape == g.shape
        kernel_inputs = [g, b, lr, momentum, dampening, weight_decay, int(nesterov), t, params]

        wp.launch(
            kernel=self.ndim_kernels[i],
            dim=params.shape,
            inputs=kernel_inputs,
            device=params.device,
        )
