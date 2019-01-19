from swervedrive.icr import Controller
from swervedrive.icr.kinematicmodel import KinematicModel
import numpy as np
import math


def unlimited_rotation_controller(
    beta_dot_bounds, beta_2dot_bounds, phi_dot_bounds, phi_2dot_bounds
):
    alpha = np.array([0, math.pi / 2, math.pi, math.pi * 3 / 4]).reshape(-1, 1)  # modules_alpha
    c = Controller(
        alpha,
        np.array([[1]] * 4),  # modules_l
        np.array([[0]] * 4),  # modules_b
        np.array([[0.1]] * 4),  # modules_r
        np.array([0] * 3),  # epsilon_init
        [-2 * math.pi, 2 * math.pi],  # beta_bounds
        [-0.5, 0.5],  # beta_dot_bounds
        [-1e6, 1e6],  # beta_2dot_bounds
        [-1e6, 1e6],  # phi_dot_bounds
        [-1e6, 1e6],  # phi_2dot_bounds
    )
    c.kinematic_model.state = KinematicModel.State.RUNNING
    c._beta_offsets = np.array(c.alpha - np.full((4, 1), math.pi / 2))
    return c


def build_controller(lower_bounds, upper_bounds):
    c = unlimited_rotation_controller(
        *[[l, u] for l, u in zip(lower_bounds, upper_bounds)]
    )
    return c
