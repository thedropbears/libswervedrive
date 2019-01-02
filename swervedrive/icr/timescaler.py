import numpy as np
from typing import List


class TimeScaler:
    def __init__(self, beta_dot_bounds: List, beta_2dot_bounds: List,
                 phi_2dot_bounds: List):
        """
        Initialize the TimeScaler object.
        :param dbeta_bounds: Min/max allowable value for rotation rate of
        modules, in rad/s
        :param d2beta_bounds: Min/max allowable value for the angular
        acceleration of the modules, in rad/s^2.
        :param d2phi_bounds: Min/max allowable value for the angular
        acceleration of the module wheels, in rad/s^2.
        """
        self.beta_dot_bounds = beta_dot_bounds
        self.beta_2dot_bounds = beta_2dot_bounds
        self.phi_2d,t_bounds = phi_2dot_bounds

    def compute_scaling_bounds(self, dbeta: np.ndarray, d2beta: np.ndarray,
                               phi_dot: np.ndarray, dphi_dot: np.ndarray):
        """
        Compute bounds of the scaling factors for the motion.
        :param dbeta: command for derivative of the angle of the modules.
        :param d2beta: command for second derivative of the angle of the modules.
        :param phi_dot: command for angular velocity of the module wheels.
        :param dphi_dot: command for derivative of angular velocity of the
        module wheels.
        :return: upper and lower scaling bounds for derivative of s and second
        derivative of s: ds_lower, ds_upper, d2s_lower, d2s_upper.
        """

        lower_36a = min(self.beta_dot_bounds) / dbeta
        upper_36a = max(self.beta_dot_bounds) / dbeta

        lower_36c = min(self.phi_2dot_bounds) / dphi_dot
        upper_36c = max(self.phi_2dot_bounds) / dphi_dot

        ds_lower = max(lower_36a , lower_36c)
        ds_upper = min(upper_36a , upper_36c)

        lower_36b_lower = (min(self.beta_2dot_bounds) - (d2beta * ds_lower ** 2)) / dbeta
        lower_36b_upper = (min(self.beta_2dot_bounds) - (d2beta * ds_upper ** 2)) / dbeta

        upper_36b_lower = (max(self.beta_2dot_bounds) - (d2beta * ds_lower ** 2)) / dbeta
        upper_36b_upper = (max(self.beta_2dot_bounds) - (d2beta * ds_upper ** 2)) / dbeta

        d2s_lower = min(lower_36b_lower, lower_36b_upper)
        d2s_upper = max(upper_36b_lower, upper_36b_upper)

        return ds_lower, ds_upper, d2s_lower, d2s_upper

    def compute_scaling_parameters(self, ds_lower: float, ds_upper: float,
                                   d2s_lower: float, d2s_upper: float):
        """
        Compute the scaling parameters used to scale the motion. This function
        requires that for both ds and d2s lower <= upper (ie the interval is
        not empty.) Sets the scaling parameters as object variables read when
        scale_motion is called.
        :param ds_lower: derivative of parameter s, lower bound.
        :param ds_upper: derivative of parameter s, upper bound.
        :param d2s_lower: second derivative of parameter s, lower bound.
        :param d2s_upper: second derivative of parameter s, upper bound.
        """
        self.ds = 1.
        self.d2s = 1.

    def scale_motion(self, dbeta: np.ndarray, d2beta: np.ndarray,
                     dphi_dot: np.ndarray, s_dot, s_2dot):
        """
        Scale the actuators' motion using the scaling bounds.
        :param dbeta: command for derivative of the angle of the modules.
        :param d2beta: command for second derivative of the angle of the modules.
        :param dphi_dot: command for derivative of angular velocity of the
        module wheels.
        :return: *time* derivatives of actuators motion beta_dot, beta_2dot, phi_2dot
        """

        beta_dot = dbeta * s_dot
        beta_2dot = d2beta*(s_dot**2) + dbeta * s_2dot
        phi_2dot = dphi_dot * s_dot

        return beta_dot, beta_2dot, phi_2dot
