from .estimator import Estimator
from .pathplanner import PathPlanner
from .kinematicmodel import KinematicModel
from .timescaler import TimeScaler
from .motionintegrator import MotionIntegrator
import numpy as np
from typing import List


class Controller:
    """
    Controller is the class that implements the control algorithm and
    instantiates all of the supporting classes.
    Note on notation: "d" in front of a variable name indicates it's
    derivative (notated as a dash in the papers). variable_dot indicates that
    this is the *time* derivative in a variable, indicated by a dot above the
    symbol in the paper. d2 or 2dot indicates second derivative.
    """
    def __init__(self, modules_alpha: np.ndarray, modules_l: np.ndarray,
                 modules_b: np.ndarray, modules_r: List, epsilon_init: np.ndarray, beta_bounds: List,
                 beta_dot_bounds: List, beta_2dot_bounds: List,
                 phi_dot_bounds: List, phi_2dot_bounds: List):
        """
        Initialize the Estimator object. The order in the following arrays
        must be preserved throughout all arguments passed to this object.
        :param modules_alpha: array containing the angle to each of the modules,
        measured counter clockwise from the x-axis.
        :param modules_l: distance to the axis of rotation of each module from
        the origin of the chassis frame
        :param modules_b: distance from the axis of rotation of each module to
        it's contact with the ground.
        :param epsilon_init: Initial epsilon value (position) of the robot.
        :param beta_bounds: Min/max allowable value for steering angle, in rad.
        :param beta_dot_bounds: Min/max allowable value for rotation rate of
        modules, in rad/s
        :param beta_2dot_bounds: Min/max allowable value for the angular
        acceleration of the modules, in rad/s^2.
        :param phi_dot_bounds: Min/max allowable value for rotation rate of
        module wheels, in rad/s
        :param phi_2dot_bounds: Min/max allowable value for the angular
        acceleration of the module wheels, in rad/s^2.
        """
        self.alpha = modules_alpha
        self.l = modules_l
        self.b = modules_b
        self.r = modules_r
        self.n_modules = len(self.alpha)

        self.icre = Estimator(epsilon_init, self.alpha, self.l, self.b)

        self.path_planner = PathPlanner(self.alpha, self.l, phi_dot_bounds,
                                        k_lmda=1, k_mu=1)
        self.kinematic_model = KinematicModel(self.alpha, self.l, self.b, k_beta=1)
        self.scaler = TimeScaler(beta_dot_bounds, beta_2dot_bounds, phi_2dot_bounds)
        self.integrator = MotionIntegrator(beta_bounds, phi_dot_bounds, self.b, self.r)

    def control_step(self, modules_beta: np.ndarray, modules_phi_dot: np.ndarray,
                     lmda_d: np.ndarray, mu_d: float, delta_t: float):
        """
        Perform a control step.
        :param modules_beta: Measured angles for each module.
        :param modules_phi_dot: Measured angular velocity for each wheel.
        :param lmda_d: desired ICR.
        :param mu_d: desired mu (rotation about ICR).
        :param delta_t: time over which control step will be executed.
        :return: beta_c, phi_dot_c
        """
        lmda_e = self.icre.estimate_lmda(modules_beta)
        mu_e = self.icre.estimate_mu(modules_phi_dot, lmda_e)
        xi_e = self.icre.compute_odometry(lmda_e, mu_e, delta_t)

        k_b = 1
        backtrack = True

        while backtrack:
            dlmda, d2lmda, dmu = self.path_planner.compute_chassis_motion(
                lmda_d, lmda_e, mu_d, mu_e, k_b
            )

            dbeta, d2beta, phi_dot_p, dphi_dot_p = \
                self.kinematic_model.compute_actuators_motion(dlmda, d2lmda, dmu)

            s_dot_l, s_dot_u, s_2dot_l, s_2dot_u = \
                self.scaler.compute_scaling_bounds(dbeta, d2beta, dphi_dot_p)

            if s_dot_l <= s_dot_u and s_2dot_l <= s_2dot_u:
                backtrack = False
            else:
                k_b = self.update_backtracking_parameter(k_b)

        self.scaler.compute_scaling_parameters(
            s_dot_l, s_dot_u, s_2dot_l, s_2dot_u)
        beta_dot, beta_2dot, phi_2dot_p = self.scaler.scale_motion(
            dbeta, d2beta, dphi_dot_p)

        beta_c, phi_dot_c = self.integrator.integrate_motion(
            beta_dot, beta_2dot, phi_dot_p, phi_2dot_p, delta_t)

        return beta_c, phi_dot_c, xi_e

        # return np.zeros(shape=(self.n_modules,)), np.zeros(shape=(self.n_modules,))
