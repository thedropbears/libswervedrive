from swervedrive.icr.kinematicmodel import KinematicModel

import math
import numpy as np


def test_compute_actuators_motion():
    alpha = np.array([0, math.pi / 2, math.pi * 3 / 4, math.pi])
    l = np.array([1.0] * 4)
    b = np.array([0.0] * 4)
    r = np.array([0.1] * 4)
    k_beta = 1.0

    km = KinematicModel(alpha, l, b, r, k_beta)

    lmda = np.array([0, 0, 1])
    lmda_dot = np.array([0, 0, 1])
    lmda_2dot = np.array([0, 0, -1])
    mu = 1.0
    mu_dot = 1.0

    beta_prime, beta_2prime, phi_dot, phi_dot_prime = km.compute_actuators_motion(
        lmda, lmda_dot, lmda_2dot, mu, mu_dot
    )

    """
    We do the calculations by hand to check that this is correct
    Consider the first wheel
    alpha = 0
    a = [1 0 0]T
    a_orth = [0 1 0]T
    l = [0 0 1]T
    s1_lmda = [0 0 sin(1)].[1 0 -1] - [1 1 cos(1)].[0 1 0]
        = [0 0 -sin(1)] - [0 1 0]
        = [0 -1 -sin(1)]
        ~ [0 -1 -0.84]
    s2_lmda = [1 1 cos(1)].[1 0 -1] + [1 1 sin(1)].[0 1 0]
        = [1 0 -cos(1)] + [0 1 0]
        ~ [1 1 -0.54]
    beta_prime = -[0 -1 -0.84].[0 0 1]/[1 1 -0.54].[0 0 1]
        = - -0.84/-0.54
        ~ -1.557
    beta_2prime = (-2*-1.557*[1 1 -0.54].[0 0 1]+[0 -1 -0.84].[0 0 -1]) /
        [1 1 -0.54].[0 0 1]
        = (3.115*-0.54+0.84)/-0.54
        ~ 4.672
    phi_dot = ([1 1 -0.540]-[0 0 0]).[0 0 1]*1-0*-1.557)/0.1
        = -0.540/0.1
        ~ -5.40
    phi_dot_prime = ([1 1 -0.540]-[0 0 0]).([0 0 1]*1+[0 0 1]*1)-0*4.672)/0.1
        = ([1 1 -0.54].[0 0 2])/0.1
        ~ -10.8
    """
    assert np.isclose(beta_prime[0], -1.557, atol=1e-2)
    assert np.isclose(beta_2prime[0], 4.672, atol=1e-2)
    assert np.isclose(phi_dot[0], -5.4, atol=1e-2)
    assert np.isclose(phi_dot_prime[0], -10.8, atol=1e-2)
