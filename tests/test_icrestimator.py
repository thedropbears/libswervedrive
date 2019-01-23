import numpy as np
import math
import pytest
from swervedrive.icr import Estimator
from swervedrive.icr.estimator import shortest_distance
from swervedrive.icr.kinematicmodel import cartesian_to_lambda as c2l

from hypothesis import given, example
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

global tolerance
tolerance = 0.01


def init_icre(alphas, ls):
    alphas = np.array(alphas).reshape(-1, 1)
    ls = np.array(ls).reshape(-1, 1)
    epsilon = np.zeros(shape=(3, 1))
    icre = Estimator(epsilon, alphas, ls)
    return icre


def test_estimate_lambda_manual():
    icre = init_icre([0, math.pi / 2, math.pi], [1, 1, 1])

    q = np.array([[0],[0],[0]])  # ICR on the robot's origin
    desired_lmda = c2l(0, 0)
    lmda_e = icre.estimate_lmda(q)
    assert np.allclose(desired_lmda, lmda_e, atol=tolerance)

    q = np.array([[math.pi / 4], [0], [-math.pi / 4]])
    desired_lmda = c2l(0, -1)
    lmda_e = icre.estimate_lmda(q)
    assert np.allclose(desired_lmda, lmda_e, atol=tolerance)

    # driving along the y axis
    q = np.array([[0], [math.pi / 2], [0]])
    # so the ICR should be on the U axis
    desired_lmda = c2l(1e20, 0)  # Rotation about infinity on x-axis
    lmda_e = icre.estimate_lmda(q)
    assert np.allclose(desired_lmda, lmda_e, atol=tolerance)

    # A square robot with 4 wheels, one at each corner, the difference between each
    # alpha value is pi/4 and the distance from the centre of the robot to each module
    # is the same
    alphas = np.arange(4) * math.pi / 2
    icre = init_icre(alphas, [1] * 4)

    # test case from the simulator
    q = np.array([[0.0], [0.0], [math.pi], [math.pi]])
    desired_lmda = np.array([[0], [0], [1]])
    lmda_e = icre.estimate_lmda(q)
    assert np.allclose(desired_lmda, lmda_e, atol=tolerance)

    # ICR on a wheel, should be a singularity
    q = np.array([[-math.pi / 4], [0], [math.pi / 4], [0]])
    desired_lmda = c2l(0, 1)
    lmda_e = icre.estimate_lmda(q)
    assert np.allclose(desired_lmda, lmda_e, atol=tolerance)
    singularity, wheel = icre.handle_singularities(lmda_e)
    # assert singularity
    # assert wheel == 1

    # Another square robot with side length of 2 to make calculations simpler
    alphas += math.pi / 4
    icre = init_icre(alphas, [math.sqrt(2)] * 4)
    # ICR on one side of the robot frame between wheels 1 and 2
    q = np.array(
        [
            [-(math.atan(2 / 1) - math.pi / 4)],
            [-math.pi / 4],
            [math.pi / 4],
            [math.atan(2 / 1) - math.pi / 4],
        ]
    )
    desired_lmda = c2l(-1, 0)
    lmda_e = icre.estimate_lmda(q)
    assert np.allclose(desired_lmda, lmda_e, atol=tolerance)

    # # afaik this is the worst case scenario, 2 wheel singularities and a co-linear singularity
    # q = np.array([-math.pi/4, math.pi/4, math.pi/4, -math.pi/4])
    # icr = np.array([0.5, 0.5, 1/math.sqrt(2)])
    # desired_lmda = icr * 1 / np.linalg.norm(icr)
    # lmda_e = icre.estimate_lmda(q)
    # assert np.allclose(desired_lmda, lmda_e.T, atol=tolerance)

@given(
        lmda=arrays(np.float, (1,3), elements=st.floats(min_value=0, max_value=1)),
        lmda_sign=st.floats(min_value=-1, max_value=1))
@example(lmda=np.array([[0.89442719, 0.0, 0.4472136]]), lmda_sign=-0.0)
def test_estimate_lambda(lmda, lmda_sign):
    icre = init_icre([0, math.pi / 2, math.pi], [1, 1, 1])
    if np.linalg.norm(lmda) == 0:
        return
    lmda = (lmda / np.linalg.norm(lmda)).reshape(-1, 1)
    lmda_signed = math.copysign(1, lmda_sign) * lmda
    q = icre.S(lmda_signed)
    lmda_e = icre.estimate_lmda(q)
    assert abs(lmda.T.dot(lmda_e)) > 1-tolerance, "Actual: %s\nEstimate: %s\nBetas: %s" % (lmda, lmda_e, q)

@given(
        lmda=arrays(np.float, (1,3), elements=st.floats(min_value=0, max_value=1)),
        lmda_sign=st.floats(min_value=-1, max_value=1),
        errors=arrays(np.float, (1,3), elements=st.floats(min_value=-math.pi*2/180,
            max_value=math.pi*2/180))
        )
def test_estimate_lambda_under_uncertainty(lmda, lmda_sign, errors):
    icre = init_icre([0, math.pi / 2, math.pi], [1, 1, 1])
    if np.linalg.norm(lmda) == 0:
        return
    lmda = (lmda / np.linalg.norm(lmda)).reshape(-1, 1)
    lmda_signed = math.copysign(1, lmda_sign) * lmda
    q = icre.S(lmda_signed) + errors.reshape(-1,1)
    lmda_e = icre.estimate_lmda(q)
    q_e = icre.S(lmda_e)
    d = shortest_distance(q, q_e)
    assert np.isclose(d, 0, atol=math.pi*4/180).all(), "Actual: %s\nEstimate: %s\nBeta errors: %s" % (lmda, lmda_e, errors/math.pi*180)


def test_joint_space_conversion():
    icre = init_icre([math.pi / 4], [1])
    lmda = np.array([0, 0, -1]).reshape(-1, 1)
    beta_target = np.array([0]).reshape(-1, 1)
    assert np.allclose(beta_target, icre.S(lmda))
    lmda = np.array([0, -1, 0]).reshape(-1, 1)
    beta_target = np.array([math.pi / 4]).reshape(-1, 1)
    assert np.allclose(beta_target, icre.S(lmda))

    # square robot with side length of 2 to make calculations simpler
    alpha = math.pi / 4
    alphas = [alpha, math.pi - alpha, -math.pi + alpha, -alpha]
    icre = init_icre(alphas, [math.sqrt(2)] * 4)

    icr = np.array([-1, 0, 1]).reshape(-1, 1)
    lmda = icr * 1 / np.linalg.norm(icr)
    beta_target = np.array(
        [
            -math.acos(6 / (2 * math.sqrt(10))),
            -math.pi / 4,
            math.pi / 4,
            math.acos(6 / (2 * math.sqrt(10))),
        ]
    ).reshape(-1, 1)
    assert np.allclose(beta_target, icre.S(lmda), atol=tolerance)


def test_solve():
    # for now, check only for runtime errors until compute_derivatives works
    icre = init_icre([math.pi / 4, -math.pi / 4, math.pi], [1, 1, 1])
    lmda = np.array([0, -1, 0]).reshape(-1, 1)
    S_u = np.array([1 / math.sqrt(2), 1 / math.sqrt(2), 0])
    S_v = np.array([0, 0, 1])
    S_w = None
    q = np.array([0, 0, 0]).reshape(-1, 1)
    icre.solve(S_u, S_v, S_w, q, lmda)


def test_compute_derivatives():
    # for now, check only for runtime errors
    icre = init_icre(
        [0, math.pi / 2, math.pi, math.pi * 3 / 4], [1, 1, 1, 1]
    )
    lmda = np.array([0, 0, -1]).reshape(-1, 1)
    S_u, S_v, S_w = icre.compute_derivatives(lmda)


def test_handle_singularities():
    icre = init_icre([0, math.pi / 2, math.pi], [1, 1, 1])
    # icr on wheel 0 on the R^2 plane
    icr = np.array([1, 0, 1]).reshape(-1, 1)
    lmda = icr * 1 / np.linalg.norm(icr)
    singularity, wheel_number = icre.handle_singularities(lmda)
    assert singularity
    assert wheel_number is 0
    icr = np.array([100, 0, 1]).reshape(-1, 1)
    lmda = icr * 1 / np.linalg.norm(icr)
    singularity, wheel_number = icre.handle_singularities(lmda)
    assert not singularity
    assert wheel_number is None


def test_update_parameters():
    icre = init_icre([0, math.pi / 2, math.pi], [1, 1, 1])
    q = np.array([[0],[0],[0]])  # ICR on the robot's origin
    desired_lmda = np.array([[0], [0], [1]])
    u, v = -0.7, -0.7  # ICR estimate both negative
    lmda_estimate = np.array([[u], [v], [math.sqrt(1 - np.linalg.norm([u, v]))]])
    delta_u, delta_v, delta_w = 0.7, 0.7, None
    lmda_t, worse = icre.update_parameters(lmda_estimate, delta_u, delta_v, delta_w, q)
    assert np.allclose(lmda_t, desired_lmda)
    assert not worse
    delta_u, delta_v, delta_w = -0.7, -0.7, None
    lmda_t, worse = icre.update_parameters(lmda_estimate, delta_u, delta_v, delta_w, q)
    assert worse, lmda_t


def test_select_starting_points():
    icre = init_icre([0, math.pi / 2, math.pi], [1, 1, 1])
    q = np.array([[0]]*3)  # ICR on the robot's origin
    desired_lmda = np.array([[0], [0], [-1]])
    starting_points = icre.select_starting_points(q)
    for sp in starting_points:
        assert np.allclose(desired_lmda[:2], sp[:2])

    q = np.array([[math.pi / 4], [0], [-math.pi / 4]])
    icr = np.array([[0], [-1], [1]])
    desired_lmda = icr * 1 / np.linalg.norm(icr)
    starting_points = icre.select_starting_points(q)
    assert np.allclose(desired_lmda[:2], starting_points[0][:2])
    # should have unit norm
    for sp in starting_points:
        assert np.isclose(np.linalg.norm(sp), 1)

    # driving along the y axis
    q = np.array([[0], [math.pi / 2], [0]])
    # so the ICR should be on the U axis
    desired_lmda = np.array([[1], [0], [0]])
    starting_points = icre.select_starting_points(q)
    assert np.allclose(desired_lmda[:2], (starting_points[0][:2]))
    for sp in starting_points:
        assert np.isclose(np.linalg.norm(sp), 1)

    # A square robot with 4 wheels, one at each corner, the difference between each
    # alpha value is pi/4 and the distance from the centre of the robot to each module
    # is the same
    # test case from the simulator
    alpha = math.pi / 4
    alphas = [alpha, math.pi - alpha, -math.pi + alpha, -alpha]
    icre = init_icre(alphas, [1] * 4)

    q = np.array([[0], [0], [math.pi], [math.pi]])
    desired_lmda = np.array([[0], [0], [1]])
    sp = icre.select_starting_points(q)
    close = []
    for p in sp:
        close.append(np.allclose(desired_lmda, p, atol=tolerance))
    assert any(close)

    # Another square robot with side length of 2 to make calculations simpler
    alpha = math.pi / 4
    alphas = [alpha, math.pi - alpha, -math.pi + alpha, -alpha]
    icre = init_icre(alphas, [math.sqrt(2)] * 4)
    # Two wheels are pointing in the same direction (the lines between them are co-linear), the other
    # two perpendiculars meet at a point halfway between the first two wheel. - currently failing
    q = np.array(
        [
            [-math.acos(6 / (2 * math.sqrt(10)))],
            [-math.pi / 4],
            [math.pi / 4],
            [math.acos(6 / (2 * math.sqrt(10)))],
        ]
    )
    icr = np.array([[-1], [0], [1]])
    desired_lmda = icr * 1 / np.linalg.norm(icr)
    sp = icre.select_starting_points(q)
    close = []
    for p in sp:
        close.append(np.allclose(desired_lmda, p, atol=tolerance))
    assert any(close)


def test_shortest_distance_manual():
    from swervedrive.icr.estimator import shortest_distance

    def check_aligned(a, b):
        assert abs(a.dot(b)[0,0]/(np.linalg.norm(a)*np.linalg.norm(b))) - 1 < tolerance
    # S_lmda on robot origin
    q = np.array([[2 * math.pi], [7 * math.pi], [math.pi / 2], [math.pi]])
    S_lmda = np.array([[0]] * 4)
    check_aligned(shortest_distance(q, S_lmda),
                  np.array([[0], [0], [-math.pi / 2], [0]]).T)

    q = np.array([[-2 * math.pi], [-7 * math.pi], [-math.pi / 2], [-math.pi]])
    S_lmda = np.array([[0]] * 4)
    check_aligned(shortest_distance(q, S_lmda),
                  np.array([[0], [0], [-math.pi / 2], [0]]).T)

@given(
        q1=arrays(np.float, (1,3), elements=st.floats(min_value=-math.pi/2, max_value=math.pi/2-0.01)),
        q2=arrays(np.float, (1,3), elements=st.floats(min_value=-math.pi/2, max_value=math.pi/2-0.01)),
)
@example(q1=np.array([[0.01745329], [0.01745329], [0.01745329]]),
        q2=np.array([[0],[0],[0]]))
def test_shortest_distance(q1, q2):
    from swervedrive.icr.estimator import shortest_distance
    diff = shortest_distance(q1.reshape(-1,1), q2.reshape(-1,1))
    assert all(diff <= math.pi/2)
    assert all(diff >= -math.pi/2)
    for qi, qj, d in zip(q1[:,0], q2[:,0], diff[:,0]):
        if qi != qj:
            assert d != 0, "q1: %s\nq2: %s\nshortest dist: %s" % (q1,q2,diff)
