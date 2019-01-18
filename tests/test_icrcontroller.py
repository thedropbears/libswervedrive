import math
import numpy as np

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from .helpers import unlimited_rotation_controller, build_controller


def cartesian_to_lambda(x, y):
    return np.reshape(1 / np.linalg.norm([x, y, 1]) * np.array([x, y, 1]), (3, 1))


def test_icrc_init():
    c = unlimited_rotation_controller(
        [-0.5, 0.5], [-1e-6, 1e-6], [-1e-6, 1e-6], [-1e-6, 1e-6]
    )
    assert c is not None


def assert_velocity_bounds(c, delta_beta, phi_dot_cmd, dt):
    # Check limits are respected
    tol = 1e-16  # to ensure we don't go over due to a floating point error
    assert all([(db) >= (c.beta_dot_bounds[0] * dt) - tol for db in delta_beta])
    assert all([(db) <= (c.beta_dot_bounds[1] * dt) + tol for db in delta_beta])
    assert all((pc) >= (c.phi_dot_bounds[0]) - tol for pc in phi_dot_cmd)
    assert all((pc) <= (c.phi_dot_bounds[1]) + tol for pc in phi_dot_cmd)


@given(
    lmda_d=arrays(np.float, (1, 3), elements=st.floats(min_value=1e-6, max_value=1)),
    lmda_d_sign=st.floats(
        min_value=-1, max_value=1
    ),  # make this *float* to give uniform distribution
    lower_bounds=st.lists(st.floats(-10, -1e-6), min_size=4, max_size=4),
    upper_bounds=st.lists(st.floats(1e-6, 10), min_size=4, max_size=4),
)
def test_respect_velocity_bounds(lmda_d, lmda_d_sign, lower_bounds, upper_bounds):
    # could do this using st.builds but that does not provide a view into what
    # the values that caused the failure were which is useful for diagnosis
    c = build_controller(lower_bounds, upper_bounds)

    # Modules can only rotate at a maximum of 0.5 rad/s
    # Make sure the controller respects these limits
    iterations = 0
    modules_beta = np.array([[0]] * 4)
    modules_phi_dot = np.array([[0]] * 4)
    lmda_d = math.copysign(1, lmda_d_sign) * lmda_d / np.linalg.norm(lmda_d)

    mu_d = 1.0
    dt = 0.1

    beta_prev = modules_beta
    phi_dot_prev = modules_phi_dot
    while iterations < 50:
        beta_cmd, phi_dot_cmd, xi_e = c.control_step(
            beta_prev, phi_dot_prev, lmda_d, mu_d, dt
        )

        delta_beta = beta_cmd - beta_prev
        assert_velocity_bounds(c, delta_beta, phi_dot_cmd, dt)

        beta_prev = beta_cmd
        phi_dot_prev = phi_dot_cmd
        iterations += 1

    lmda_e = c.icre.estimate_lmda(beta_prev)
    mu_e = c.kinematic_model.estimate_mu(phi_dot_prev, lmda_e)
    assert np.allclose(lmda_e, lmda_d, atol=1e-2) or np.allclose(
        -lmda_e, lmda_d, atol=1e-2
    ), "Controller did not reach target"
    assert np.isclose(mu_e, mu_d, atol=1e-2) or np.isclose(-mu_e, mu_d, atol=1e-2)


def test_structural_singularity_command():
    c = unlimited_rotation_controller(
        [-0.5, 0.5], [-1e-6, 1e-6], [-1e-6, 1e-6], [-1e-6, 1e-6]
    )
    # test to see what happens if we place the ICR on a structural singularity
    iterations = 0
    modules_beta = np.array([0] * 4)
    modules_phi_dot = np.array([0] * 4)
    lmda_d_normal = np.array([1, 0, 0])

    lmda_singularity = cartesian_to_lambda(
        c.l[0] * math.sin(c.alpha[0]), c.l[1] * math.sin(c.alpha[1])
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

        beta_cmd, phi_dot_cmd, xi_e = c.control_step(
            beta_prev, phi_dot_prev, lmda_d, mu_d, dt
        )

        # check that our estimated ICR never gets close to the structural singularity,
        # despite being the setpoint
        lmda_e = c.icre.estimate_lmda(beta_cmd).reshape(3)
        assert not all(np.isclose(lmda_e, lmda_singularity, atol=1e-2))

        beta_prev = beta_cmd
        phi_dot_prev = phi_dot_cmd
        iterations += 1
