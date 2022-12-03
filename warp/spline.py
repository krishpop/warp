import warp as wp
import numpy as np

def make_spline_functions(dim, vector):
    if dim == 1:
        def spline_coefficients(
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=float),
            u: wp.array(dtype=float),
                coeffs: wp.array(dtype=float)):
            """
            Compute the coefficients of a cubic spline.
            :param xs: the x values
            :param ys: the y values
            :return: the coefficients
            """
            u[0] = 0.0
            for i in range(1, num_xs-1):
                sig = (xs[i] - xs[i-1]) / (xs[i+1] - xs[i-1])
                p = sig * coeffs[i-1] + 2.0
                coeffs[i] = (sig - 1.0) / p
                u[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]) - \
                    (ys[i] - ys[i-1]) / (xs[i] - xs[i-1])
                u[i] = (6.0 * u[i] / (xs[i+1] - xs[i-1]) -
                        sig * u[i-1]) / p
            for k in range(num_xs-2, -1, -1):
                coeffs[k] = coeffs[k] * coeffs[k+1] + u[k]
    else:
        def spline_coefficients(
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            u: wp.array(dtype=vector),
                coeffs: wp.array(dtype=vector)):
            """
            Compute the coefficients of a cubic spline.
            :param xs: the x values
            :param ys: the y values
            :return: the coefficients
            """
            u[0] = vector_from_scalar(0.0)
            for i in range(1, num_xs-1):
                sig = (xs[i] - xs[i-1]) / (xs[i+1] - xs[i-1])
                p = coeffs[i-1] * sig + 2.0
                s = sig - 1.0
                coeffs[i] = wp.cw_div(vector_from_scalar(s), p)
                u[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]) - \
                    (ys[i] - ys[i-1]) / (xs[i] - xs[i-1])
                u[i] = (u[i] * 6.0 / (xs[i+1] - xs[i-1]) -
                        u[i-1] * sig)
                u[i] = wp.cw_div(u[i], p)
            for k in range(num_xs-2, -1, -1):
                coeffs[k] = wp.cw_mul(coeffs[k], coeffs[k+1]) + u[k]

    # create a new module for this spline dimension
    module = wp.get_module(f"spline_{dim}d")
    # XXX use namespace to avoid name collisions with other spline dimensions
    namespace = f"spline_{dim}d"
    if dim > 1:
        # register generic vector constructor
        if dim == 2:
            def vector_from_scalar(x: float):
                return wp.vec2(x)
        elif dim == 3:
            def vector_from_scalar(x: float):
                return wp.vec3(x)
        elif dim == 4:
            def vector_from_scalar(x: float):
                return wp.vec4(x)
        vector_from_scalar = wp.Function(vector_from_scalar, key="vector_from_scalar", namespace=namespace)
        module.register_function(vector_from_scalar)
        
    coefficients_kernel = wp.Kernel(
        func=spline_coefficients, key=f"spline_{dim}d_coefficients_kernel", module=module)

    def evaluate(
            x: float,
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector)) -> vector:
        """
        Evaluate the spline at a given point.
        """
        # find the interval
        i = int(0)
        for j in range(num_xs - 1):
            if x > xs[j + 1]:
                i += 1
        h = xs[i + 1] - xs[i]
        a = (xs[i + 1] - x) / h
        b = (x - xs[i]) / h
        # evaluate the spline
        y = a * ys[i] + b * ys[i + 1] + \
            ((a ** 3. - a) * coeffs[i] + (b ** 3. - b)
             * coeffs[i + 1]) * h ** 2. / 6.
        return y

    evaluate = wp.Function(evaluate, key="evaluate", namespace=namespace)
    module.register_function(evaluate)

    def evaluate_kernel(
            x: wp.array(dtype=float),
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            interpolated: wp.array(dtype=vector)):
        i = wp.tid()
        interpolated[i] = evaluate(x[i], num_xs, xs, ys, coeffs)

    evaluate_kernel = wp.Kernel(
        func=evaluate_kernel, key=f"spline_{dim}d_evaluate_kernel", module=module)

    def evaluate_d(
            x: float,
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector)) -> vector:
        """
        Evaluate the spline at a given point.
        """
        # find the interval
        i = int(0)
        for j in range(num_xs - 1):
            if x > xs[j + 1]:
                i += 1
        h = xs[i + 1] - xs[i]
        u = xs[i]
        v = xs[i + 1]
        s = ys[i]
        t = ys[i + 1]
        a = coeffs[i]
        b = coeffs[i + 1]
        c3 = (b - a)/(6. * h)
        c2 = (a * v - b * u)/(2. * h)
        c1 = (a * h**2. - 3. * a * v**2. - b * h**2. +
              3. * b * u**2. - 6. * s + 6. * t)/(6. * h)
        # evaluate
        return c1 + 2.*c2*x + 3.*c3*x**2.

    evaluate_d = wp.Function(evaluate_d, key="evaluate_d", namespace=namespace)
    module.register_function(evaluate_d)
    # module.functions["evaluate_d"] = evaluate_d

    def evaluate_d_kernel(
            x: wp.array(dtype=float),
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            interpolated: wp.array(dtype=vector)):
        i = wp.tid()
        interpolated[i] = evaluate_d(x[i], num_xs, xs, ys, coeffs)

    evaluate_d_kernel = wp.Kernel(
        func=evaluate_d_kernel, key=f"spline_{dim}d_evaluate_d_kernel", module=module)

    def evaluate_dd(
            x: float,
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector)) -> vector:
        """
        Evaluate the spline at a given point.
        """
        # find the interval
        i = int(0)
        for j in range(num_xs - 1):
            if x > xs[j + 1]:
                i += 1
        h = xs[i + 1] - xs[i]
        u = xs[i]
        v = xs[i + 1]
        a = coeffs[i]
        b = coeffs[i + 1]
        c3 = (b - a)/(6. * h)
        c2 = (a * v - b * u)/(2. * h)
        # evaluate
        return 2.*c2 + 6.*c3*x

    evaluate_dd = wp.Function(evaluate_dd, key="evaluate_dd", namespace=namespace)
    module.register_function(evaluate_dd)

    def evaluate_dd_kernel(
            x: wp.array(dtype=float),
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            interpolated: wp.array(dtype=vector)):
        i = wp.tid()
        interpolated[i] = evaluate_dd(x[i], num_xs, xs, ys, coeffs)

    evaluate_dd_kernel = wp.Kernel(
        func=evaluate_dd_kernel, key=f"spline_{dim}d_evaluate_dd_kernel", module=module)

    if dim == 1:
        def closest_coord(
                point: vector,
                num_xs: int,
                xs: wp.array(dtype=float),
                ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
                gd_max_iter: int,
                gd_step_size: float) -> float:
            """
            Find the coordinate of the closest point on the spline to a given point.
            """
            best_t = 0.5 * (xs[0] + xs[1])
            dir = evaluate(best_t, num_xs, xs, ys, coeffs) - point
            best_d = dir * dir
            # TODO prune the search using bounding boxes of spline segments?
            for i in range(num_xs - 1):
                x1 = xs[i]
                x2 = xs[i + 1]
                # starting point for gradient descent
                t = float(0.5 * (x1 + x2))
                step_size = gd_step_size * (x2 - x1)
                # find local minimum of square distance between point and spline segment i
                for iter in range(gd_max_iter):
                    # derivative of square distance between point at t and spline
                    s = evaluate(t, num_xs, xs, ys, coeffs)
                    ds = evaluate_d(t, num_xs, xs, ys, coeffs)
                    df = 2. * s * ds - 2. * point * ds
                    t -= df * step_size
                    t = wp.clamp(t, x1, x2)
                # evaluate square distance between point and spline at t
                dir = evaluate(t, num_xs, xs, ys, coeffs) - point
                d = dir * dir
                if d < best_d:
                    best_t = t
                    best_d = d
            return best_t
    else:
        def closest_coord(
                point: vector,
                num_xs: int,
                xs: wp.array(dtype=float),
                ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
                gd_max_iter: int,
                gd_step_size: float) -> float:
            """
            Find the coordinate of the closest point on the spline to a given point.
            """
            best_t = 0.5 * (xs[0] + xs[1])
            dir = evaluate(best_t, num_xs, xs, ys, coeffs) - point
            best_d = wp.dot(dir, dir)
            # TODO prune the search using bounding boxes of spline segments?
            for i in range(num_xs - 1):
                x1 = xs[i]
                x2 = xs[i + 1]
                # starting point for gradient descent
                t = float(0.5 * (x1 + x2))
                step_size = gd_step_size * (x2 - x1)
                # find local minimum of square distance between point and spline segment i
                for iter in range(gd_max_iter):
                    # derivative of square distance between point at t and spline
                    s = evaluate(t, num_xs, xs, ys, coeffs)
                    ds = evaluate_d(t, num_xs, xs, ys, coeffs)
                    df = 2. * wp.dot(s, ds) - 2. * wp.dot(point, ds)
                    t -= df * step_size
                    t = wp.clamp(t, x1, x2)
                # evaluate square distance between point and spline at t
                dir = evaluate(t, num_xs, xs, ys, coeffs) - point
                d = wp.dot(dir, dir)
                if d < best_d:
                    best_t = t
                    best_d = d
            return best_t

    closest_coord = wp.Function(closest_coord, key="closest_coord", namespace=namespace)
    module.register_function(closest_coord)

    def closest_coord_kernel(
            points: wp.array(dtype=vector),
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            gd_max_iter: int,
            gd_step_size: float,
            coords: wp.array(dtype=float)):
        i = wp.tid()
        coords[i] = closest_coord(
            points[i], num_xs, xs, ys, coeffs, gd_max_iter, gd_step_size)

    closest_coord_kernel = wp.Kernel(
        func=closest_coord_kernel, key=f"spline_{dim}d_closest_coord_kernel", module=module)

    def closest_point(
            point: vector,
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            gd_max_iter: int,
            gd_step_size: float) -> vector:
        """
        Find the closest_point from a point to the spline.
        """
        t = closest_coord(point, num_xs, xs, ys, coeffs,
                             gd_max_iter, gd_step_size)
        s = evaluate(t, num_xs, xs, ys, coeffs)
        return s

    closest_point = wp.Function(closest_point, key="closest_point", namespace=namespace)
    module.register_function(closest_point)

    def closest_point_kernel(
            points: wp.array(dtype=vector),
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            gd_max_iter: int,
            gd_step_size: float,
            closest_points: wp.array(dtype=vector)):
        i = wp.tid()
        closest_points[i] = closest_point(
            points[i], num_xs, xs, ys, coeffs, gd_max_iter, gd_step_size)

    closest_point_kernel = wp.Kernel(
        func=closest_point_kernel, key=f"spline_{dim}d_closest_point_kernel", module=module)

    def distance(
            point: vector,
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            gd_max_iter: int,
            gd_step_size: float) -> float:
        """
        Find the distance from a point to the spline.
        """
        s = closest_point(point, num_xs, xs, ys, coeffs,
                             gd_max_iter, gd_step_size)
        return wp.length(s - point)

    distance = wp.Function(distance, key="distance", namespace=namespace)
    module.register_function(distance)

    def distance_kernel(
            points: wp.array(dtype=vector),
            num_xs: int,
            xs: wp.array(dtype=float),
            ys: wp.array(dtype=vector),
            coeffs: wp.array(dtype=vector),
            gd_max_iter: int,
            gd_step_size: float,
            distances: wp.array(dtype=float)):
        i = wp.tid()
        distances[i] = distance(
            points[i], num_xs, xs, ys, coeffs, gd_max_iter, gd_step_size)

    distance_kernel = wp.Kernel(
        func=distance_kernel, key=f"spline_{dim}d_distance_kernel", module=module)

    return (
        coefficients_kernel,
        evaluate,
        evaluate_kernel,
        evaluate_d,
        evaluate_d_kernel,
        evaluate_dd,
        evaluate_dd_kernel,
        closest_coord,
        closest_coord_kernel,
        closest_point,
        closest_point_kernel,
        distance,
        distance_kernel
    )


(
    spline_1d_coefficients_kernel,
    spline_1d_evaluate,
    spline_1d_evaluate_kernel,
    spline_1d_evaluate_d,
    spline_1d_evaluate_d_kernel,
    spline_1d_evaluate_dd,
    spline_1d_evaluate_dd_kernel,
    spline_1d_closest_coord,
    spline_1d_closest_coord_kernel,
    spline_1d_closest_point,
    spline_1d_closest_point_kernel,
    spline_1d_distance,
    spline_1d_distance_kernel
) = make_spline_functions(1, float)

(
    spline_2d_coefficients_kernel,
    spline_2d_evaluate,
    spline_2d_evaluate_kernel,
    spline_2d_evaluate_d,
    spline_2d_evaluate_d_kernel,
    spline_2d_evaluate_dd,
    spline_2d_evaluate_dd_kernel,
    spline_2d_closest_coord,
    spline_2d_closest_coord_kernel,
    spline_2d_closest_point,
    spline_2d_closest_point_kernel,
    spline_2d_distance,
    spline_2d_distance_kernel
) = make_spline_functions(2, wp.vec2)

(
    spline_3d_coefficients_kernel,
    spline_3d_evaluate,
    spline_3d_evaluate_kernel,
    spline_3d_evaluate_d,
    spline_3d_evaluate_d_kernel,
    spline_3d_evaluate_dd,
    spline_3d_evaluate_dd_kernel,
    spline_3d_closest_coord,
    spline_3d_closest_coord_kernel,
    spline_3d_closest_point,
    spline_3d_closest_point_kernel,
    spline_3d_distance,
    spline_3d_distance_kernel
) = make_spline_functions(3, wp.vec3)

(
    spline_4d_coefficients_kernel,
    spline_4d_evaluate,
    spline_4d_evaluate_kernel,
    spline_4d_evaluate_d,
    spline_4d_evaluate_d_kernel,
    spline_4d_evaluate_dd,
    spline_4d_evaluate_dd_kernel,
    spline_4d_closest_coord,
    spline_4d_closest_coord_kernel,
    spline_4d_closest_point,
    spline_4d_closest_point_kernel,
    spline_4d_distance,
    spline_4d_distance_kernel
) = make_spline_functions(4, wp.vec4)


class Spline:
    """
    A class for cubic spline interpolation.
    """
    vector_types = [float, wp.vec2, wp.vec3, wp.vec4]
    coeff_kernels = [
        spline_1d_coefficients_kernel,
        spline_2d_coefficients_kernel,
        spline_3d_coefficients_kernel,
        spline_4d_coefficients_kernel]
    eval_kernels = [
        [spline_1d_evaluate_kernel, spline_2d_evaluate_kernel,
         spline_3d_evaluate_kernel, spline_4d_evaluate_kernel],
        [spline_1d_evaluate_d_kernel, spline_2d_evaluate_d_kernel,
         spline_3d_evaluate_d_kernel, spline_4d_evaluate_d_kernel],
        [spline_1d_evaluate_dd_kernel, spline_2d_evaluate_dd_kernel,
         spline_3d_evaluate_dd_kernel, spline_4d_evaluate_dd_kernel]]
    distance_kernels = [
        spline_1d_distance_kernel,
        spline_2d_distance_kernel,
        spline_3d_distance_kernel,
        spline_4d_distance_kernel]

    def __init__(self, xs, ys):
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")

        if isinstance(xs, np.ndarray):
            self.xs = wp.array(xs, dtype=wp.float32,
                               device=wp.get_preferred_device())
        else:
            self.xs = xs
        if isinstance(ys, np.ndarray):
            if ys.ndim == 1 or ys.shape[1] == 1:
                self.ys = wp.array(ys, dtype=wp.float32,
                                   device=wp.get_preferred_device())
                self.point_dim = 1
            else:
                assert ys.shape[1] < len(
                self.vector_types), "ys must have 4 or fewer dimensions"
                self.ys = wp.array(
                    ys, dtype=self.vector_types[ys.shape[1]-1], device=wp.get_preferred_device())
                self.point_dim = ys.shape[1]
        else:
            self.ys = ys
            if ys.dtype in (wp.float32, wp.float64):
                self.point_dim = 1
            elif ys.dtype == wp.vec2:
                self.point_dim = 2
            elif ys.dtype == wp.vec3:
                self.point_dim = 3
            elif ys.dtype == wp.vec4:
                self.point_dim = 4
            else:
                raise ValueError(
                    "only float, vec2, vec3, and vec4 are supported dtypes for ys")
        u = wp.empty_like(self.ys)
        self.coeffs = wp.zeros_like(self.ys)
        wp.launch(self.coeff_kernels[self.point_dim-1],
                  dim=1,
                  inputs=[len(self.xs), self.xs, self.ys, u],
                  outputs=[self.coeffs],
                  device=self.xs.device)

    @property
    def num_xs(self):
        return len(self.xs)

    @property
    def device(self):
        return self.xs.device

    def get_coefficients(self, i):
        """
        Computes the polynomial coefficients for the cubic spline segment starting at node i.
        Returns [c0, c1, c2, c3] for a cubic spline f(x) = c0*x^3 + c1*x^2 + c2*x + c3.
        """
        xs = self.xs.numpy()
        ys = self.ys.numpy()
        cs = self.coeffs.numpy()
        h = xs[i + 1] - xs[i]
        u = xs[i]
        v = xs[i + 1]
        s = ys[i]
        t = ys[i + 1]
        a = cs[i]
        b = cs[i + 1]
        c0 = (b - a)/(6 * h)
        c1 = (a * v - b * u)/(2 * h)
        c2 = (a * h**2 - 3 * a * v**2 - b * h**2 +
              3 * b * u**2 - 6 * s + 6 * t)/(6 * h)
        c3 = (-a * h**2 * v + a * v**3 + b * h**2 * u -
              b * u**3 + 6 * s * v - 6 * t * u)/(6 * h)
        return (c0, c1, c2, c3)

    def __call__(self, x, order=0):
        assert 0 <= order <= 2, "order of derivative must be between 0 and 2"
        if isinstance(x, float):
            # modify query point in place
            query = wp.array([x], device=self.ys.device)
            wp.launch(
                self.eval_kernels[order][self.point_dim-1],
                dim=1,
                inputs=[query, len(self.xs), self.xs, self.ys, self.coeffs],
                outputs=[query],
                device=self.xs.device)
            return query.numpy()[0]
        elif isinstance(x, (list, tuple, np.ndarray)):
            query = wp.array(x, device=self.ys.device, dtype=wp.float32)
        else:
            assert isinstance(x, wp.array), "unsupported type"
            query = x
        result = wp.empty(
            len(query),
            device=self.ys.device,
            dtype=self.ys.dtype,
            requires_grad=query.requires_grad)
        wp.launch(
            self.eval_kernels[order][self.point_dim-1],
            dim=len(query),
            inputs=[query, len(self.xs), self.xs, self.ys, self.coeffs],
            outputs=[result],
            device=self.xs.device)
        if isinstance(x, (list, tuple, np.ndarray)):
            return result.numpy()
        else:
            return result

    def distance(self, p, gd_max_iter=100, gd_step_size=1e-1):
        """
        Returns the distance from the spline to the point(s) p.
        """
        single_point = False
        if isinstance(p, (list, tuple, np.ndarray)):
            if len(p) == self.point_dim and isinstance(p[0], float):
                single_point = True
                query = wp.array([p], device=self.ys.device,
                                 dtype=self.ys.dtype)
            else:
                assert len(
                    p[0]) == self.point_dim, f"point must be a {self.point_dim}D vector"
                query = wp.array(p, device=self.ys.device, dtype=self.ys.dtype)
        else:
            query = p
        result = wp.empty(
            len(query),
            device=self.ys.device,
            dtype=self.xs.dtype,
            requires_grad=query.requires_grad)
        wp.launch(
            self.distance_kernels[self.point_dim-1],
            dim=len(query),
            inputs=[query, len(self.xs), self.xs, self.ys,
                    self.coeffs, gd_max_iter, gd_step_size],
            outputs=[result],
            device=self.xs.device)
        if single_point:
            return result.numpy()[0]
        elif isinstance(p, (list, tuple, np.ndarray)):
            return result.numpy()
        else:
            return result

    def plot(self):
        import matplotlib.pyplot as plt
        xs = self.xs.numpy()
        ys = self.ys.numpy()
        fine_xs = np.linspace(xs[0], xs[-1], 500)
        fine_ys = self(fine_xs)
        fig, ax = plt.subplots(nrows=self.point_dim, ncols=1)
        for i in range(self.point_dim):
            ax[i].plot(xs, ys[:, i], 'o')
            ax[i].plot(fine_xs, fine_ys[:, i])
            ax[i].grid()
        plt.show()