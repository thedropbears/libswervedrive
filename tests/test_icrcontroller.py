from swervedrive.icr import Controller
from swervedrive.icr.kinematicmodel import KinematicModel

import math
import numpy as np
import pytest


def cartesian_to_lambda(x, y):
    return np.reshape(1 / np.linalg.norm([x, y, 1]) * np.array([x, y, 1]), (3, 1))


@pytest.fixture
def unlimited_rotation_controller():
    c = Controller(
        np.array([0, math.pi / 2, math.pi, math.pi * 3 / 4]),  # modules_alpha
        np.array([1] * 4),  # modules_l
        np.array([0] * 4),  # modules_b
        np.array([0.1] * 4),  # modules_r
        np.array([0] * 3),  # epsilon_init
        [-2 * math.pi, 2 * math.pi],  # beta_bounds
        [-0.5, 0.5],  # beta_dot_bounds
        [-1e6, 1e6],  # beta_2dot_bounds
        [-1e6, 1e6],  # phi_dot_bounds
        [-1e6, 1e6],  # phi_2dot_bounds
    )
    c.kinematic_model.state = KinematicModel.State.RUNNING
    return c


def test_icrc_init(unlimited_rotation_controller):
    assert unlimited_rotation_controller is not None


def assert_velocity_bounds(c, delta_beta, phi_dot_cmd, dt):
    # Check limits are respected
    tol = 1e-16 # to ensure we don't go over due to a floating point error
    assert all([
        (db) >= (c.beta_dot_bounds[0] * dt)-tol
        for db in delta_beta
    ])
    assert all([
        (db) <= (c.beta_dot_bounds[1] * dt)+tol
        for db in delta_beta
    ])
    assert all(
        (pc) >= (c.phi_dot_bounds[0])-tol
        for pc in phi_dot_cmd
    )
    assert all(
        (pc) <= (c.phi_dot_bounds[1])+tol
        for pc in phi_dot_cmd
    )


def test_respect_velocity_bounds(unlimited_rotation_controller):
    # Modules can only rotate at a maximum of 0.5 rad/s
    # Make sure the controller respects these limits
    iterations = 0
    modules_beta = np.array([0] * 4)
    modules_phi_dot = np.array([0] * 4)
    lmda_d = np.array([1, 0, 0])

    mu_d = 1.0
    dt = 0.1

    beta_prev = modules_beta
    phi_dot_prev = modules_phi_dot
    while iterations < 100:
        beta_cmd, phi_dot_cmd, xi_e = unlimited_rotation_controller.control_step(
            beta_prev, phi_dot_prev, lmda_d, mu_d, dt
        )

        delta_beta = beta_cmd - beta_prev
        assert_velocity_bounds(unlimited_rotation_controller, delta_beta, phi_dot_cmd, dt)

        beta_prev = beta_cmd
        phi_dot_prev = phi_dot_cmd
        iterations += 1
    lmda_e = unlimited_rotation_controller.icre.estimate_lmda(beta_prev)
    assert all(
        np.isclose(lmda_e.reshape(3),
                   lmda_d,
                   atol=1e-2)
    ), "Controller did not reach target"
    assert np.isclose(unlimited_rotation_controller.kinematic_model.estimate_mu(phi_dot_prev, lmda_e),
                      mu_d, atol=1e-2)


def test_structural_singularity_command(unlimited_rotation_controller):
    # test to see what happens if we place the ICR on a structural singularity
    iterations = 0
    modules_beta = np.array([0] * 4)
    modules_phi_dot = np.array([0] * 4)
    lmda_d_normal = np.array([1, 0, 0])

    lmda_singularity = cartesian_to_lambda(
        unlimited_rotation_controller.l[0]*math.sin(unlimited_rotation_controller.alpha[0]),
        unlimited_rotation_controller.l[1]*math.sin(unlimited_rotation_controller.alpha[1])
    ).reshape(3)

    mu_d = 0.1
    dt = 0.1

    beta_prev = modules_beta
    phi_dot_prev = modules_phi_dot
    while iterations < 100:
        # let the controller do its thing for a while, then put the command on a singularity
        lmda_d = lmda_d_normal
        if iterations > 20:
            lmda_d = lmda_singularity

        beta_cmd, phi_dot_cmd, xi_e = unlimited_rotation_controller.control_step(
            beta_prev, phi_dot_prev, lmda_d, mu_d, dt
        )

        # check that our estimated ICR never gets close to the structural singularity,
        # despite being the setpoint
        lmda_e = unlimited_rotation_controller.icre.estimate_lmda(beta_cmd).reshape(3)
        assert not all(
            np.isclose(lmda_e,
                       lmda_singularity,
                       atol=1e-2)
        )

        beta_prev = beta_cmd
        phi_dot_prev = phi_dot_cmd
        iterations += 1
