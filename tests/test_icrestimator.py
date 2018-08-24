import numpy as np
import math
from icrestimator import ICREstimator

def init_icre(alphas, ls, bs):
    alphas = np.array(alphas)
    ls = np.array(ls)
    bs = np.array(bs)
    icre = ICREstimator(alphas, ls, bs, 20)
    return icre

def test_joint_space_conversion():
    icre = init_icre([math.pi/4], [1], [0])
    lmda = np.array([0, 0, -1]).reshape(-1, 1)
    beta_target = np.array([0])
    assert np.allclose(beta_target, icre.S(lmda))
    lmda = np.array([0, -1, 0]).reshape(-1, 1)
    beta_target = np.array([math.pi/4])
    assert np.allclose(beta_target, icre.S(lmda))