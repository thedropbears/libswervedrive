from swervedrive.icr import Controller
from swervedrive.icr.kinematicmodel import KinematicModel

import math
import numpy as np
import pytest


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


def test_icrc_init():
    c = unlimited_rotation_controller()
    assert c is not None


def test_respect_velocity_bounds():
    c = unlimited_rotation_controller()
    # Modules can only rotate at a maximum of 0.5 rad/s
    # Make sure the controller respects these limits
    iterations = 0
    modules_beta = np.array([0] * 4)
    modules_phi_dot = np.array([0] * 4)
    lmda_d = np.array([1, 0, 0])
    mu_d = 0.0
    dt = 0.1
    beta_prev = modules_beta
    while iterations < 100:
        beta_cmd, phi_cmd, xi_e = c.control_step(
            modules_beta, modules_phi_dot, lmda_d, mu_d, dt
        )
        delta_beta = beta_cmd - beta_prev
        # Checked limits are respected
        assert all((db) >= c.beta_dot_bounds[0] * dt for db in delta_beta)
        assert all((db) <= c.beta_dot_bounds[1] * dt for db in delta_beta)
        # Check if we have hit our desired target (within some tolerance)
        if all(abs(db) < 1e-3 for db in delta_beta):
            break
        beta_prev = beta_cmd
        iterations += 1
    assert iterations < 100, "Controller did not reach target"
