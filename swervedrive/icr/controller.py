from .estimator import Estimator
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
        # TODO: reshape these to column vectors
        self.alpha = np.array(modules_alpha).reshape(-1)
        self.l = np.array(modules_l).reshape(-1)
        self.b = np.array(modules_b).reshape(-1)
        self.r = np.array(modules_r).reshape(-1)
        self.n_modules = len(self.alpha)
        self.beta_bounds = beta_bounds
        self.beta_dot_bounds = beta_dot_bounds
        self.beta_2dot_bounds = beta_2dot_bounds
        self.phi_dot_bounds = phi_dot_bounds
        self.phi_2dot_bounds = phi_2dot_bounds

        self.icre = Estimator(epsilon_init, self.alpha, self.l)

        self.kinematic_model = KinematicModel(
            self.alpha, self.l, self.b, self.r, k_beta=40
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

        assert len(modules_beta.shape) == 2 and modules_beta.shape[0] == self.n_modules, modules_beta
        assert len(modules_phi_dot.shape) == 2 and modules_phi_dot.shape[0] == self.n_modules, modules_phi_dot

        if lmda_d is not None:
            assert lmda_d.shape == (3,1), lmda_d
        if self.kinematic_model.state == KinematicModel.State.RECONFIGURING:
            # we can't simply set the estimated lmda because it is poorly defined
            # in the reconfiguring state - so we must simply discard this command
            if lmda_d is None:
                return modules_beta, modules_phi_dot, self.kinematic_model.xi
            beta_d = self.icre.S(lmda_d)
            dbeta = self.kinematic_model.reconfigure_wheels(beta_d, modules_beta)
            d2beta = np.array([[0]] * len(dbeta))
            phi_dot_c = np.array([[0]] * len(dbeta))
            dphi_dot_c = np.array([[0]] * len(dbeta))
            beta_c, phi_dot_c = self.integrate_motion(
                dbeta, d2beta, phi_dot_c, dphi_dot_c, modules_beta, delta_t
            )
            return beta_c, phi_dot_c, self.kinematic_model.xi

        lmda_e = self.icre.estimate_lmda(modules_beta)
        assert lmda_e.shape == (3,1), lmda_e
        mu_e = self.kinematic_model.estimate_mu(modules_phi_dot, lmda_e)
        if lmda_d is None:
            lmda_d = lmda_e
        xi_e = self.kinematic_model.compute_odometry(lmda_e, mu_e, delta_t)

        k_b = 1
        backtrack = True

        if self.kinematic_model.state == KinematicModel.State.STOPPING:
            mu_d = 0
            lmda_d = lmda_e

        while backtrack:
            dlmda, d2lmda, dmu = self.kinematic_model.compute_chassis_motion(
                lmda_d, lmda_e, mu_d, mu_e, k_b, self.phi_dot_bounds, k_lmda=4, k_mu=4
            )

            dbeta, d2beta, phi_dot_p, dphi_dot_p = self.kinematic_model.compute_actuators_motion(
                lmda_e, dlmda, d2lmda, mu_e, dmu
            )

            s_dot_l, s_dot_u, s_2dot_l, s_2dot_u = self.scaler.compute_scaling_bounds(
                dbeta, d2beta, dphi_dot_p
            )

            if s_dot_l <= s_dot_u and s_2dot_l <= s_2dot_u:
                backtrack = False
            else:
                # update backtracking parameter - end of section 3.4
                k_b /= 2

        self.scaler.compute_scaling_parameters(s_dot_l, s_dot_u, s_2dot_l, s_2dot_u)
        beta_dot, beta_2dot, phi_2dot_p = self.scaler.scale_motion(
            dbeta, d2beta, dphi_dot_p
        )

        beta_c, phi_dot_c = self.integrate_motion(
            beta_dot, beta_2dot, phi_dot_p, phi_2dot_p, modules_beta, delta_t
        )

        assert len(beta_c.shape) == 2 and beta_c.shape[0] == self.n_modules, beta_c
        assert len(phi_dot_c.shape) == 2 and phi_dot_c.shape[0] == self.n_modules, phi_dot_c

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
        # NOTE: the current code actually expects these shapes to be 1-d arrays, so will need to be changed.
        # I suspect that this is because the class members self.b and self.r are 1d.
        assert len(beta_dot.shape) == 2 and beta_dot.shape[0] == self.n_modules, beta_dot
        assert len(beta_2dot.shape) == 2 and beta_2dot.shape[0] == self.n_modules, beta_2dot
        assert len(phi_dot.shape) == 2 and phi_dot.shape[0] == self.n_modules, phi_dot
        assert len(phi_2dot.shape) == 2 and phi_2dot.shape[0] == self.n_modules, phi_2dot
        assert len(beta_e.shape) == 2 and beta_e.shape[0] == self.n_modules, beta_e

        # TODO: change back to using class variables once they are the correct shape
        b = np.reshape(self.b, (-1, 1))
        r = np.reshape(self.r, (-1, 1))

        phi_dot_c = (phi_dot - np.multiply(np.divide(b, r), beta_dot)) + (
            phi_2dot - np.multiply(np.divide(b, r), beta_2dot)
        ) * delta_t  # 40b
        # Check limits
        fd = 1.0  # scaling factor
        for pdc in phi_dot_c.reshape(-1, 1):
            if pdc * fd > self.phi_dot_bounds[1]:
                fd = self.phi_dot_bounds[1] / pdc
            if pdc * fd < self.phi_dot_bounds[0]:
                fd = self.phi_dot_bounds[0] / pdc
        # Also check that we respect the rotation rate limits
        delta_beta_c = beta_dot * delta_t + 1 / 2 * beta_2dot * (delta_t ** 2)
        for dbc in delta_beta_c.reshape(-1, 1):
            if dbc * fd > self.beta_dot_bounds[1] * delta_t:
                fd = self.beta_dot_bounds[1] * delta_t / dbc
            if dbc * fd < self.beta_dot_bounds[0] * delta_t:
                fd = self.beta_dot_bounds[0] * delta_t / dbc
        beta_c = (beta_e + fd * delta_beta_c).reshape(-1,1)  # 40a
        phi_dot_c = fd * phi_dot_c  # 42

        if not all(bc < self.beta_bounds[1] for bc in beta_c) or not all(
            bc > self.beta_bounds[0] for bc in beta_c
        ):
            beta_c = beta_e  # 43
            phi_dot_c = phi_dot
            # This requires a wheel reconfig in the kinematic model
            self.kinematic_model.state = KinematicModel.State.STOPPING

        return beta_c, phi_dot_c
