from swervedrive.icr.kinematicmodel import KinematicModel

import math
import numpy as np
import pytest


def cartesian_to_lambda(x, y):
    return np.reshape(1 / np.linalg.norm([x, y, 1]) * np.array([x, y, 1]), (3, 1))


@pytest.fixture
def kinematic_model():
    alpha = np.array([0, math.pi / 2, math.pi, math.pi * 3 / 2])
    l = np.array([1.0] * 4)
    b = np.array([0.0] * 4)
    r = np.array([0.1] * 4)
    k_beta = 1.0

    km = KinematicModel(alpha, l, b, r, k_beta)
    km.state = KinematicModel.State.RUNNING

    return km


def test_compute_actuators_motion(kinematic_model):
    lmda = np.array([0, 0, 1])
    lmda_dot = np.array([0, 0, 1])
    lmda_2dot = np.array([0, 0, -1])
    mu = 1.0
    mu_dot = 1.0

    beta_prime, beta_2prime, phi_dot, phi_dot_prime = kinematic_model.compute_actuators_motion(
        lmda, lmda_dot, lmda_2dot, mu, mu_dot
    )

    """
    We do the calculations by hand to check that this is correct
    Consider the first wheel
    alpha = 0
    a = [1 0 0]T
    a_orth = [0 1 0]T
    l = [0 0 1]T
    sin(beta) = [0 1 0].[0 0 1]
        = 0
    cos(beta) = ([1 0 0]-[0 0 1]).[0 0 1]
        = [1 0 -1].[0 0 1]
        = -1

    s1_lmda = sin(b)[1 0 -1] - cos(b)[0 1 0]
        = [0 0 0] - [0 1 0]
        = [0 -1 0]
    s2_lmda = cos(b)[1 0 -1] + sin(b)[0 1 0]
        = -1*[1 0 -1] + [0 0 0]
        = [-1 0 1]
    beta_prime = -[0 -1 0].[0 0 1]/[-1 0 1].[0 0 1]
        = - 0/1
        = 0
    beta_2prime = (-2*0*[-1 0 1].[0 0 1]+[0 -1 0].[0 0 -1]) /
        [-1 0 1].[0 0 1]
        = (0+0)/1
        = 0
    phi_dot = ([-1 0 1]-[0 0 0]).[0 0 1]*1-0*0)/0.1
        = 1/0.1
        = 10
    phi_dot_prime = ([-1 0 1]-[0 0 0]).([0 0 1]*1+[0 0 1]*1)-0*0)/0.1
        = ([-1 0 1].[0 0 2])/0.1
        = 20
    """
    assert np.isclose(beta_prime[0], 0.0, atol=1e-2)
    assert np.isclose(beta_2prime[0], 0.0, atol=1e-2)
    assert np.isclose(phi_dot[0], 10, atol=1e-2)
    assert np.isclose(phi_dot_prime[0], 20, atol=1e-2)


def test_singularity_on_wheel(kinematic_model):
    lmda = cartesian_to_lambda(1, 0)  # this is the location of the first wheel
    lmda_dot = np.array([0, 2, 1])
    lmda_2dot = np.array([0, -1, -1])
    mu = 1.0
    mu_dot = 1.0

    beta_prime, beta_2prime, phi_dot, phi_dot_prime = kinematic_model.compute_actuators_motion(
        lmda, lmda_dot, lmda_2dot, mu, mu_dot
    )
    # Wheel at ICR should have all return values (except drive accel) set to zero
    assert np.isclose(beta_prime[0], 0, atol=1e-2)
    assert np.isclose(beta_2prime[0], 0, atol=1e-2)
    assert np.isclose(phi_dot[0], 0, atol=1e-2)

    assert not np.isclose(beta_prime[1:], [0] * 3, atol=1e-2).any()
    assert not np.isclose(beta_2prime[1:], [0] * 3, atol=1e-2).any()
    assert not np.isclose(phi_dot[1:], [0] * 3, atol=1e-2).any()


def test_s_perp(kinematic_model):
    lmda = cartesian_to_lambda(0, 0)  # Centre of robot
    beta = np.arctan(
        kinematic_model.a_orth.transpose().dot(lmda)
        / (kinematic_model.a - kinematic_model.l_vector).transpose().dot(lmda)
    )
    s_lmda = kinematic_model.s_perp(lmda)
    expected_s1 = np.array([[0, -1, 0, 1], [1, 0, -1, 0], [0, 0, 0, 0]])
    expected_s2 = np.array([[-1, 0, 1, 0], [0, -1, 0, 1], [1, 1, 1, 1]])
    assert np.isclose(s_lmda[0], expected_s1).all()
    assert np.isclose(s_lmda[1], expected_s2).all()


def test_estimate_mu(kinematic_model):
    lmda = cartesian_to_lambda(0, 0)  # Centre of robot
    expected = 1.0  # rad/s
    phi_dot = np.array([expected * 1.0] * 4)
    mu = kinematic_model.estimate_mu(phi_dot, lmda)
    assert np.isclose(
        mu, expected * kinematic_model.r[0], atol=0.01
    ).all(), "\nCalculated mu: %f,\nexpected: %f\nlambda: %s" % (mu, expected, lmda)

    lmda = cartesian_to_lambda(2, 0)  # Right of robot
    expected = 1.0  # rad/s
    phi_dot = np.array(
        [-expected * 1.0, -expected * 5.0 ** 0.5, expected * 3.0, expected * 5.0 ** 0.5]
    )
    mu = kinematic_model.estimate_mu(phi_dot, lmda)
    assert np.isclose(mu, expected * kinematic_model.r[0], atol=0.01).all(), (
        "\nCalculated mu: %f,\nexpected: %f\nlambda: %s"
        % (mu, expected * kinematic_model.r[0], lmda)
    )


def test_compute_odometry(kinematic_model):
    x_dot = 1.0
    y_dot = 1.0
    theta_dot = 0.0
    mu = np.linalg.norm([x_dot, y_dot, theta_dot])
    lmda = np.array([-y_dot, x_dot, theta_dot]) / mu
    dt = 0.1

    xi = kinematic_model.compute_odometry(lmda, mu, dt)
    assert np.isclose(xi, np.array([x_dot * dt, y_dot * dt, 0.0]), atol=1e-2).all()

    # Reset the odometry
    kinematic_model.xi[0, 0] = kinematic_model.xi[0, 1] = 0.0
    kinematic_model.xi[0, 2] = math.pi / 2

    xi = kinematic_model.compute_odometry(lmda, mu, dt)
    assert np.isclose(
        xi, np.array([-x_dot * dt, y_dot * dt, math.pi / 2]), atol=1e-2
    ).all()

    # Reset the odometry
    kinematic_model.xi[0, 0] = kinematic_model.xi[0, 1] = 0.0
    kinematic_model.xi[0, 2] = math.pi / 2

    x_dot = 1.0
    y_dot = 0.1
    theta_dot = 1.0
    mu = np.linalg.norm([x_dot, y_dot, theta_dot])
    lmda = np.array([-y_dot, x_dot, theta_dot]) / mu
    xi = kinematic_model.compute_odometry(lmda, mu, dt)
    xi = kinematic_model.compute_odometry(lmda, mu, dt)
    assert xi[0, 0] < x_dot * dt * 2
    assert xi[0, 1] > y_dot * dt * 2
