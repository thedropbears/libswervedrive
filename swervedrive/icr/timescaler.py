import numpy as np
from typing import List


class TimeScaler:

    # we ignore the corresponding constraints if their governing command
    # value is close to zero
    # TODO: figure out what the tolerances should be
    ignore_beta_thresh: float = 1e-2
    ignore_phi_thresh: float = 1e-2

    def __init__(
        self, beta_dot_bounds: List, beta_2dot_bounds: List, phi_2dot_bounds: List
    ):
        """
        Initialize the TimeScaler object.
        :param dbeta_bounds: Min/max allowable value for rotation rate of
            modules, in rad/s
        :param d2beta_bounds: Min/max allowable value for the angular
            acceleration of the modules, in rad/s^2.
        :param d2phi_bounds: Min/max allowable value for the angular
            acceleration of the module wheels, in rad/s^2.
        """
        self.beta_dot_b = beta_dot_bounds
        self.beta_2dot_b = beta_2dot_bounds
        self.phi_2dot_b = phi_2dot_bounds

    def compute_scaling_bounds(self, dbeta: np.ndarray, d2beta: np.ndarray,
                               dphi_dot: np.ndarray):
        """
        Compute bounds of the scaling factors for the motion across all modules
        This function effectively computes bounds such that the possibly
        arbitrarily high amplitude of the commands from the kinematic model
        obeys the physical constraints of the robot motion.
        :param dbeta: command for derivative of the angle of the modules.
        :param d2beta: command for second derivative of the angle of the modules.
        :param dphi_dot: command for derivative of angular velocity of the
            module wheels.
        :returns: upper and lower scaling bounds for 1st and 2nd time derivatives
            of s: s_dot_l, s_dot_u, s_2dot_l, s_2dot_u
        """
        s_dot_l = 0
        s_dot_u = 1
        s_2dot_l = 0
        s_2dot_u = 1
        n_modules = len(dbeta)
        for i in range(n_modules):
            sdl, sdu = self.compute_module_s_dot_bounds(
                dbeta[i], d2beta[i], dphi_dot[i]
            )
            s_dot_l = max(s_dot_l, sdl)
            s_dot_u = min(s_dot_u, sdu)
        for i in range(n_modules):
            s2dl, s2du = self.compute_module_s_2dot_bounds(
                dbeta[i], d2beta[i], s_dot_u)
            s_2dot_l = max(s_2dot_l, s2dl)
            s_2dot_u = min(s_2dot_u, s2du)

        return s_dot_l, s_dot_u, s_2dot_l, s_2dot_u

    def compute_module_s_dot_bounds(
        self, dbeta: float, d2beta: float, dphi_dot: float
    ):
        """
        Compute bounds of the scaling factors for the motion for one module
        This function effectively computes bounds such that the possibly
        arbitrarily high amplitude of the commands from the kinematic model
        obeys the physical constraints of the robot motion.
        :param dbeta: command for derivative of the angle of the modules.
        :param d2beta: command for second derivative of the angle of the modules.
        :param dphi_dot: command for derivative of angular velocity of the
            module wheels.
        :returns: upper and lower scaling bounds for 1st time derivative
        of s: s_dot_l, s_dot_u
        """
        if (
            in_range(dbeta, self.beta_dot_b)
            and in_range(d2beta, self.beta_2dot_b)
            and in_range(dphi_dot, self.phi_2dot_b)
        ):
            # constraits in 35a and c are satisfied, no scaling required
            return 0, 1

        ignore_beta = np.isclose(dbeta, 0, atol=self.ignore_beta_thresh)
        ignore_phi = np.isclose(dphi_dot, 0, atol=self.ignore_phi_thresh)

        s_dot_l, s_dot_u = 0, 1

        if not ignore_beta:
            # need to reverse inequality if we have a negative
            (lower, upper) = (1, 0) if dbeta < 0 else (0, 1)
            # equation 36a in control paper
            s_dot_l = max(s_dot_l, self.beta_dot_b[lower] / dbeta)
            s_dot_u = min(s_dot_u, self.beta_dot_b[upper] / dbeta)
        if not ignore_phi:
            (lower, upper) = (1, 0) if dphi_dot < 0 else (0, 1)
            # equation 36c in control paper
            s_dot_l = max(s_dot_l, self.phi_2dot_b[lower] / dphi_dot)
            s_dot_u = min(s_dot_u, self.phi_2dot_b[upper] / dphi_dot)

        return s_dot_l, s_dot_u

    def compute_module_s_2dot_bounds(self, dbeta: float, d2beta: float, s_dot: float):

        if (in_range(dbeta, self.beta_dot_b)
                and in_range(d2beta, self.beta_2dot_b)):
            # constraits in 35aand c are satisfied, no scaling required
            return 0, 1

        s_2dot_l, s_2dot_u = 0, 1

        ignore_beta = np.isclose(dbeta, 0, atol=self.ignore_beta_thresh)

        if ignore_beta:
            return 0, 1

        # apply constraint on second derivative
        # must calculate here as it depends on the value of s_dot, which
        # in turn is defined by the value of s_dot_u
        (lower, upper) = (1, 0) if dbeta < 0 else (0, 1)
        s_2dot_l = max(
            s_2dot_l, (self.beta_2dot_b[lower] - d2beta * (s_dot ** 2)) / dbeta
        )
        s_2dot_u = min(
            s_2dot_u, (self.beta_2dot_b[upper] - d2beta * (s_dot ** 2)) / dbeta
        )

        return s_2dot_l, s_2dot_u

    def compute_scaling_parameters(
        self, s_dot_l: float, s_dot_u: float, s_2dot_l: float, s_2dot_u: float
    ):
        """
        Compute the scaling parameters used to scale the motion. This function
        assumes that for both ds and d2s lower <= upper (ie the interval is
        not empty.) Sets the scaling parameters as object variables read when
        scale_motion is called.
        :param s_dot_l: derivative of parameter s, lower bound.
        :param s_dot_u: derivative of parameter s, upper bound.
        :param s_2dot_l: second derivative of parameter s, lower bound.
        :param s_2dot_u: second derivative of parameter s, upper bound.
        """
        self.s_dot = s_dot_u
        self.s_2dot = s_2dot_u

    def scale_motion(self, dbeta: np.ndarray, d2beta: np.ndarray, dphi_dot: np.ndarray):
        """
        Scale the actuators' motion using the scaling bounds.
        :param dbeta: command for derivative of the angle of the modules.
        :param d2beta: command for second derivative of the angle of the modules.
        :param dphi_dot: command for derivative of angular velocity of the
            module wheels.
        :returns: *time* derivatives of actuators motion beta_dot, beta_2dot, phi_2dot
        """

        beta_dot = dbeta * self.s_dot
        beta_2dot = d2beta * (self.s_dot ** 2) + dbeta * self.s_2dot
        phi_2dot = dphi_dot * self.s_dot

        return beta_dot, beta_2dot, phi_2dot


def in_range(value, rng):
    """ Check if value(s) are in the range of list[0], list[1] """
    return (rng[0] <= value).all() and (value <= rng[1]).all()
