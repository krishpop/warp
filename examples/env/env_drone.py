# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Drone environment
#
# Simulation of a quadrotor drone with custom propeller dynamics.
# This example implements a simulation plugin for the SemiImplicitIntegrator
# that computes the propeller forces and torques.
#
###########################################################################

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType, RenderMode


air_density = wp.constant(1.225)  # kg / m^3


class PropellerData:
    def __init__(
        self,
        thrust: float = 0.109919,
        power: float = 0.040164,
        diameter: float = 0.2286,
        height: float = 0.01,
        max_rpm: float = 6396.667,
        turning_cw: bool = True,
    ):
        """
        Creates an object to store propeller information. Uses default settings of the GWS 9X5 propeller.

        Args:
        thrust: Thrust coefficient.
        power: Power coefficient.
        diameter: Propeller diameter in meters, default is for DJI Phantom 2.
        height: Height of cylindrical area in meters when propeller rotates.
        max_rpm: Maximum RPM of propeller.
        turning_cw: True if propeller turns clockwise, False if counter-clockwise.
        """
        self.thrust = thrust
        self.power = power
        self.diameter = diameter
        self.height = height
        self.max_rpm = max_rpm
        self.turning_direction = 1.0 if turning_cw else -1.0

        # compute max thrust and torque
        revolutions_per_second = max_rpm / 60
        max_speed = revolutions_per_second * wp.TAU  # radians / sec
        self.max_speed_square = max_speed ** 2

        nsquared = revolutions_per_second ** 2
        self.max_thrust = self.thrust * air_density * nsquared * self.diameter ** 4
        self.max_torque = self.power * air_density * nsquared * self.diameter ** 5 / wp.TAU


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    force = prop.dir * prop.max_thrust * control
    torque = prop.dir * prop.max_torque * control * prop.turning_direction
    torque += wp.cross(prop.pos, force)
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(torque, force))


class DroneSimulationPlugin(wp.sim.SemiImplicitIntegratorPlugin):
    def __init__(self):
        super().__init__()
        self.props = []
        self.props_wp = None
        self.prop_speed = None

    def add_propeller(
        self,
        body: int,
        pos: wp.vec3,
        dir: wp.vec3,
        prop_data: PropellerData,
    ):
        """
        Add a propeller to the scene.
        """
        prop = Propeller()
        prop.body = body
        prop.pos = pos
        prop.dir = wp.normalize(dir)
        for k, v in prop_data.__dict__.items():
            setattr(prop, k, v)
        self.props.append(prop)

    def on_init(self, model, integrator):
        self.props_wp = wp.array(self.props, dtype=Propeller, device=model.device)

    def before_integrate(self, model, state_in, state_out, dt, requires_grad):
        assert self.props_wp is not None, "DroneSimulationPlugin not initialized"
        wp.launch(
            compute_prop_wrenches,
            dim=len(self.props),
            inputs=[self.props_wp, state_in.prop_control],
            outputs=[state_in.body_f],
            device=model.device,
        )

    def augment_state(self, model, state):
        # add state vector for the propeller control inputs
        state.prop_control = wp.zeros(len(self.props), dtype=float, device=model.device)


@wp.kernel
def update_prop_rotation(
    prop_rotation: wp.array(dtype=float),
    prop_control: wp.array(dtype=float),
    prop_shape: wp.array(dtype=int),
    props: wp.array(dtype=Propeller),
    dt: float,
    shape_transform: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    prop = props[tid]
    speed = prop_control[tid] * prop.max_speed_square / 2000.0  # a bit slower for better rendering
    wp.atomic_add(prop_rotation, tid, prop.turning_direction * speed * dt)
    shape = prop_shape[tid]
    shape_transform[shape] = wp.transform(prop.pos, wp.quat_from_axis_angle(prop.dir, prop_rotation[tid]))


@wp.kernel
def drone_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target: wp.vec3,
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    pos_drone = wp.transform_get_translation(body_q[env_id])
    drone_cost = wp.length_sq(pos_drone - target)

    vel_drone = body_qd[env_id]

    # encourage zero velocity
    vel_cost = wp.length_sq(vel_drone)

    wp.atomic_add(cost, env_id, drone_cost + 0.02 * vel_cost)


class DroneEnvironment(Environment):
    sim_name = "env_drone"

    num_envs = 100

    opengl_render_settings = dict(scaling=3.0, draw_axis=False)
    usd_render_settings = dict(scaling=100.0)

    # use_graph_capture = False

    sim_substeps_euler = 16
    sim_substeps_xpbd = 5

    activate_ground_plane = False

    drone_crossbar_length = 0.2
    drone_crossbar_height = 0.01
    drone_crossbar_width = 0.01

    integrator_type = IntegratorType.EULER

    controllable_dofs = [0, 1, 2, 3]
    control_gains = [1.0] * 4
    control_limits = [(0.0, 1.0)] * 4

    flight_target = wp.vec3(0.0, 0.5, 1.0)

    def __init__(self):
        self.drone_plugin = DroneSimulationPlugin()
        self.euler_settings["plugins"] = [self.drone_plugin]
        super().__init__()

        # arrays to keep track of propeller rotation just for rendering
        self.prop_rotation = None

    def setup(self, builder):
        self.prop_shapes = []
        for i in range(self.num_envs):
            xform = wp.transform(self.env_offsets[i], wp.quat_identity())
            drone = builder.add_body(name=f"drone_{i}", origin=xform)
            builder.add_shape_box(
                drone,
                hx=self.drone_crossbar_length,
                hy=self.drone_crossbar_height,
                hz=self.drone_crossbar_width)
            builder.add_shape_box(
                drone,
                hx=self.drone_crossbar_width,
                hy=self.drone_crossbar_height,
                hz=self.drone_crossbar_length)

            def add_prop(pos, cw=True):
                normal = wp.vec3(0.0, 1.0, 0.0)
                prop_data = PropellerData(turning_cw=cw)
                self.drone_plugin.add_propeller(drone, pos, normal, prop_data)
                # add fake propeller geometry
                prop_shape = builder.add_shape_box(
                    drone,
                    pos=pos,
                    hx=prop_data.diameter / 2.0,
                    hy=prop_data.diameter / 25.0,
                    hz=prop_data.diameter / 15.0,
                    density=0.0)
                self.prop_shapes.append(prop_shape)

            add_prop(wp.vec3(self.drone_crossbar_length, 0.0, 0.0), False)
            add_prop(wp.vec3(-self.drone_crossbar_length, 0.0, 0.0))
            add_prop(wp.vec3(0.0, 0.0, self.drone_crossbar_length))
            add_prop(wp.vec3(0.0, 0.0, -self.drone_crossbar_length), False)

    def before_simulate(self):
        if self.render_mode == RenderMode.OPENGL:
            self.prop_shape = wp.array(self.prop_shapes, dtype=int, device=self.device)
            self.prop_rotation = wp.zeros(len(self.prop_shapes), dtype=float, device=self.device)

    def custom_update(self):
        self.state.prop_control.fill_(wp.sin(self.sim_time) * 1e-4 + 0.187)

    def custom_render(self, render_state):
        for i in range(self.num_envs):
            self.renderer.render_sphere(
                f"target_{i}", self.flight_target + self.env_offsets[i], wp.quat_identity(), radius=0.01)

        if self.render_mode == RenderMode.OPENGL:
            # directly animate shape instances in renderer because model.shape_transform is not considered
            # by the OpenGLRenderer online
            if self.renderer._wp_instance_transforms is None:
                return
            wp.launch(
                update_prop_rotation,
                dim=len(self.prop_shapes),
                inputs=[self.prop_rotation, render_state.prop_control,
                        self.prop_shape, self.drone_plugin.props_wp, self.frame_dt],
                outputs=[self.renderer._wp_instance_transforms],
                device=self.device)

    @property
    def control(self):
        # overwrite control property to actuate propellers, not joints
        return self.state.prop_control

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array):
        wp.launch(
            drone_cost,
            dim=self.num_envs,
            inputs=[state.body_q, state.body_qd, self.flight_target],
            outputs=[cost],
            device=self.device
        )


if __name__ == "__main__":
    run_env(DroneEnvironment)
