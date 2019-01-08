from .estimator import Estimator
from .pathplanner import PathPlanner
from .kinematicmodel import KinematicModel
from .timescaler import TimeScaler
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

    def __init__(
        self,
        modules_alpha: np.ndarray,
        modules_l: np.ndarray,
        modules_b: np.ndarray,
        modules_r: np.ndarray,
        epsilon_init: np.ndarray,
        beta_bounds: List,
        beta_dot_bounds: List,
        beta_2dot_bounds: List,
        phi_dot_bounds: List,
        phi_2dot_bounds: List,
    ):
        """
        Initialize the Estimator object. The order in the following arrays
        must be preserved throughout all arguments passed to this object.
        :param modules_alpha: array containing the angle to each of the modules,
        measured counter clockwise from the x-axis.
        :param modules_l: distance to the axis of rotation of each module from
        the origin of the chassis frame
        :param modules_b: distance from the axis of rotation of each module to
        it's contact with the ground.
        :param modules_r: radii of the wheels (m).
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
        self.beta_bounds = beta_bounds
        self.beta_dot_bounds = beta_dot_bounds
        self.beta_2dot_bounds = beta_2dot_bounds
        self.phi_dot_bounds = phi_dot_bounds
        self.phi_2dot_bounds = phi_2dot_bounds

        self.icre = Estimator(epsilon_init, self.alpha, self.l, self.b)

        self.path_planner = PathPlanner(
            self.alpha, self.l, phi_dot_bounds, k_lmda=1, k_mu=1
        )
        self.kinematic_model = KinematicModel(
            self.alpha, self.l, self.b, self.r, k_beta=1
        )
        self.scaler = TimeScaler(beta_dot_bounds, beta_2dot_bounds, phi_2dot_bounds)

    def control_step(
        self,
        modules_beta: np.ndarray,
        modules_phi_dot: np.ndarray,
        lmda_d: np.ndarray,
        mu_d: float,
        delta_t: float,
    ):
        """
        Perform a control step.
        :param modules_beta: Measured angles for each module.
        :param modules_phi_dot: Measured angular velocity for each wheel.
        :param lmda_d: desired ICR.
        :param mu_d: desired mu (rotation about ICR).
        :param delta_t: time over which control step will be executed.
        :returns: beta_c, phi_dot_c, xi_e
        """
        lmda_e = self.icre.estimate_lmda(modules_beta)
        mu_e = self.icre.estimate_mu(modules_phi_dot, lmda_e)
        xi_e = self.icre.compute_odometry(lmda_e, mu_e, delta_t)

        k_b = 1
        backtrack = True

        if self.kinematic_model.state == KinematicModel.State.STOPPING:
            mu_d = 0
            lmda_d = lmda_e

        while backtrack:
            dlmda, d2lmda, dmu = self.path_planner.compute_chassis_motion(
                lmda_d, lmda_e, mu_d, mu_e, k_b
            )

            dbeta, d2beta, phi_dot_p, dphi_dot_p = self.kinematic_model.compute_actuators_motion(
                lmda_e, dlmda, d2lmda, mu_e, dmu
            )
            if self.kinematic_model.state == KinematicModel.State.RECONFIGURING:
                beta_d = self.icre.S(lmda_d)
                dbeta = self.kinematic_model.reconfigure_wheels(beta_d, modules_beta)
                d2beta = np.array([0] * len(dbeta))
                phi_dot_p = np.array([0] * len(dbeta))
                dphi_dot_p = np.array([0] * len(dbeta))

            s_dot_l, s_dot_u, s_2dot_l, s_2dot_u = self.scaler.compute_scaling_bounds(
                dbeta, d2beta, dphi_dot_p
            )

            if s_dot_l <= s_dot_u and s_2dot_l <= s_2dot_u:
                backtrack = False
            else:
                k_b = self.update_backtracking_parameter(k_b)

        self.scaler.compute_scaling_parameters(s_dot_l, s_dot_u, s_2dot_l, s_2dot_u)
        beta_dot, beta_2dot, phi_2dot_p = self.scaler.scale_motion(
            dbeta, d2beta, dphi_dot_p
        )

        beta_c, phi_dot_c = self.integrate_motion(
            beta_dot, beta_2dot, phi_dot_p, phi_2dot_p, modules_beta, delta_t
        )

        return beta_c, phi_dot_c, xi_e

    def integrate_motion(
        self,
        beta_dot: np.ndarray,
        beta_2dot: np.ndarray,
        phi_dot: np.ndarray,
        phi_2dot: np.ndarray,
        beta_e: np.ndarray,
        delta_t: float,
    ):
        """
        Integrate the motion to produce the beta and phi_dot commands.
        :param beta_dot: command for the module's angular velocity.
        :param beta_2dot: command for the module's angular acceleration.
        :param phi_dot: command for the module wheel's angular velocity.
        :param phi_2dot: command for the module wheel's angular acceleration.
        :param beta_e: the current measured beta values (angles) of the modules.
        :param delta_t: timestep over which the command will be executed.
        :returns: beta_c, phi_dot_c (module angle and wheel angular velocity
        commands)
        """

        phi_dot_c = (phi_dot - self.b / self.r * beta_dot) + (
            phi_2dot - self.b / self.r * beta_2dot
        ) * delta_t  # 40b
        # Check limits
        fd = 1.0  # scaling factor
        for pdc in phi_dot_c:
            if pdc * fd > self.phi_dot_bounds[1]:
                fd = self.phi_dot_bounds[1] / pdc
            if pdc * fd < self.phi_dot_bounds[0]:
                fd = self.phi_dot_bounds[0] / pdc
        # Also check that we respect the rotation rate limits
        delta_beta_c = beta_dot * delta_t + 1 / 2 * beta_2dot * (delta_t ** 2)
        for dbc in delta_beta_c:
            if dbc * fd > self.beta_dot_bounds[1] * delta_t:
                fd = self.beta_dot_bounds[1] * delta_t / dbc
            if dbc * fd < self.beta_dot_bounds[0] * delta_t:
                fd = self.beta_dot_bounds[0] * delta_t / dbc
        beta_c = beta_e + fd * delta_beta_c  # 40a
        phi_dot_c = fd * phi_dot_c  # 42

        if not all(bc < self.beta_bounds[1] for bc in beta_c) or not all(
            bc > self.beta_bounds[0] for bc in beta_c
        ):
            beta_c = beta_e  # 43
            phi_dot_c = phi_dot
            # This requires a wheel reconfig in the kinematic model
            self.kinematic_model.state = KinematicModel.State.STOPPING

        return beta_c, phi_dot_c
