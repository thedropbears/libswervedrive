import numpy as np
from swervedrive.icr.timescaler import TimeScaler


def assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot):
    """ Function to ensure that the inequalities in the second paper hold,
    given certain velocity/acceleeration bounds and commands. """
    scaler = TimeScaler(beta_dot_b, beta_2dot_b, phi_2dot_b)
    ds_lower, ds_upper, d2s_lower, d2s_upper = scaler.compute_scaling_bounds(
        dbeta, d2beta, dphi_dot
    )

    # inequalities are reversed for negative values
    (lower_beta, upper_beta) = (1, 0) if dbeta < 0 else (0, 1)
    (lower_phi, upper_phi) = (1, 0) if dphi_dot < 0 else (0, 1)
    ignore_beta = np.isclose(dbeta, 0, atol=0.01)
    ignore_phi = np.isclose(dphi_dot, 0, atol=0.01)

    if not ignore_beta:

        # check that we satisfy equation 36a
        assert ds_lower >= beta_dot_b[lower_beta] / dbeta
        assert ds_upper <= beta_dot_b[upper_beta] / dbeta

        # check that we satisfy equation 36b
        assert d2s_lower >= (
            (beta_2dot_b[lower_beta] - d2beta * (ds_upper ** 2)) / dbeta
        )
        assert d2s_upper <= (
            (beta_2dot_b[upper_beta] - d2beta * (ds_upper ** 2)) / dbeta
        )
    if not ignore_phi:
        # check that we satify equation 36c
        assert ds_lower >= phi_2dot_b[lower_phi] / dphi_dot
        assert ds_upper <= phi_2dot_b[upper_phi] / dphi_dot

    scaler.compute_scaling_parameters(ds_lower, ds_upper, d2s_lower, d2s_upper)
    beta_dot, beta_2dot, phi_2dot = scaler.scale_motion(dbeta, d2beta, dphi_dot)

    assert beta_dot_b[0] <= beta_dot <= beta_2dot_b[1]
    assert beta_2dot_b[0] <= beta_2dot <= beta_2dot_b[1]
    assert phi_2dot_b[0] <= phi_2dot <= beta_2dot_b[1]


def test_positive_velocities_in_range():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[0.5]]), np.array([[0.25]]), np.array([[0.25]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_negative_velocities_in_range():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[-0.5]]), np.array([[-0.25]]), np.array([[-0.25]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_positive_velocities_not_in_range():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[5]]), np.array([[1.5]]), np.array([[1.5]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_negative_velocities_not_in_range():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[-5]]), np.array([[-1.5]]), np.array([[-1.5]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_dbeta_zero():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = 0.01, -1.5, -1.5
    dbeta, d2beta, dphi_dot = np.array([[0.01]]), np.array([[-1.5]]), np.array([[-1.5]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_d2beta_zero():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[5]]), np.array([[0.01]]), np.array([[-1.5]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_dphi_dot_zero():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[-5]]), np.array([[-1.5]]), np.array([[0]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_opposing_signs():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = 5, -1.5, -5
    dbeta, d2beta, dphi_dot = np.array([[5]]), np.array([[-1.5]]), np.array([[-5]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)


def test_all_zero():
    # angular vel/accel bounds
    beta_dot_b = [-1, 1]  # rad/sec
    beta_2dot_b = [-1, 1]  # rad/sec^2
    # wheel rotation bounds
    phi_2dot_b = [-1, 1]
    # motion commands generated from the kinematic model for this timestep
    dbeta, d2beta, dphi_dot = np.array([[0]]), np.array([[0]]), np.array([[0]])
    assert_scaling_bounds(beta_dot_b, beta_2dot_b, phi_2dot_b, dbeta, d2beta, dphi_dot)
