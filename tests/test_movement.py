import math
import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from .helpers import unlimited_rotation_controller


def twist_to_icr(vx: float, vy: float, vz: float):
    """Convert a twist command (vx, vy, vz) to lmda and mu.

    Eta represents the motion about the ICR as represented in the projective plane.
    See eq.(1) of the control paper.
    """
    norm = np.linalg.norm([vx, vy, vz])
    if np.isclose(norm, 0, atol=0.01):
        return None, 0
    eta = (1 / norm) * np.array([-vy, vx, vz, norm ** 2])
    lmda = eta[0:3].reshape(-1, 1)
    mu = eta[3]
    return lmda, mu


@given(
    twist_segments=arrays(
        np.float, (3, 3), elements=st.floats(min_value=-4, max_value=4)
    )
)
@settings(deadline=1000)
def test_sequential_direction_movement(twist_segments):
    c = unlimited_rotation_controller(
        [-0.5, 0.5], [-1e1, 1e1], [-1e1, 1e1], [-1e1, 1e1]
    )
    dt = 1.0 / 20.0
    modules_beta = np.array([[0]] * 4) - c._beta_offsets
    modules_phi_dot = np.array([[0]] * 4)

    delta_beta = modules_beta
    for segment in twist_segments:
        iterations = 0
        while iterations < 20 and np.linalg.norm(delta_beta) > 1e-3:
            lmda_d, mu_d = twist_to_icr(*segment)
            beta_cmd, phi_dot_cmd, xi_e = c.control_step(
                modules_beta, modules_phi_dot, lmda_d, mu_d, dt
            )
            delta_beta = beta_cmd - modules_beta
            modules_beta = beta_cmd
            modules_phi_dot = phi_dot_cmd
            iterations += 1
        angles_robot = modules_beta + c._beta_offsets
        robot_twist = swerve_solver(
            modules_phi_dot.reshape(-1),
            angles_robot.reshape(-1),
            c.alpha.reshape(-1),
            c.l.reshape(-1),
        )
        assert np.allclose(robot_twist, segment, atol=1e-1)


def swerve_solver(module_speeds, module_angles, modules_alpha, modules_l):
    """Solve the least-squares of the speed and angles of four swerve modules
    to retrieve delta x and y in the robot frame.

    Note:
        This function uses the standard (and superior) ROS coordinate system,
        with forward being positive x, leftward being positive y, and a
        counter clockwise rotation being one about the positive z axis.
    Args:
        module_speeds: List of the speeds of each module (m/s)
        module_angles: List of the angles of each module (radians)
        module_x_offsets: Offset of each module on the x axis.
        module_y_offsets: Offset of each module on the y axis.
    Returns:
        vx: float, robot velocity on the x axis (m/s)
        vy: float, robot velocity on the y axis (m/s)
        vz: float, robot velocity about the z axis (radians/s)
    """

    module_x_offsets = np.multiply(modules_l, np.cos(modules_alpha))
    module_y_offsets = np.multiply(modules_l, np.sin(modules_alpha))

    A = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    module_states = np.zeros((8, 1), dtype=float)
    for i in range(4):
        module_dist = math.hypot(module_x_offsets[i], module_y_offsets[i])
        module_angle = math.atan2(module_y_offsets[i], module_x_offsets[i])
        A[i * 2, 2] = -module_dist * math.sin(module_angle)
        A[i * 2 + 1, 2] = module_dist * math.cos(module_angle)

        x_vel = module_speeds[i] * math.cos(module_angles[i])
        y_vel = module_speeds[i] * math.sin(module_angles[i])
        module_states[i * 2, 0] = x_vel
        module_states[i * 2 + 1, 0] = y_vel

    lstsq_ret = np.linalg.lstsq(A, module_states, rcond=None)
    vx, vy, vz = lstsq_ret[0].reshape(3)

    return np.array([vx, vy, vz])
