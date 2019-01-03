from swervedrive.icr import Controller
import numpy as np


def test_icrc_init():
    icrc = Controller(
        np.array([0]),  # modules_alpha
        np.array([1]),  # modules_l
        np.array([0.1]),  # modules_b
        np.array([0.1]),  # modules_r
        np.zeros(shape=(3, 1)),  # epsilon_init
        [-1, 1],  # beta_bounds
        [-1, 1],  # beta_dot_bounds
        [-1, 1],  # beta_2dot_bounds
        [-1, 1],  # phi_dot_bounds
        [-1, 1],  # phi_2dot_bounds
    )
