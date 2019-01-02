from typing import List
import numpy as np


class MotionIntegrator:
    def __init__(self, beta_bounds: List, phi_dot_bounds: List, modules_b, modules_r):
        """
        Initialize the MotionIntegrator class.
        :param beta_bounds: Min/max allowable value for steering angle, in rad.
        :param phi_dot_bounds: Min/max allowable value for rotation rate of
        module wheels, in rad/s
        """

        self.b = modules_b
        self.r = modules_r
        self.beta_bounds = beta_bounds
        self.phi_dot_bounds = phi_dot_bounds

    def integrate_motion(self, beta_dot: np.ndarray, beta_2dot: np.ndarray,
                         phi_dot: np.ndarray, phi_2dot: np.ndarray,
                         delta_t: float, beta_e: np.ndarray):
        """
        Integrate the motion to produce the beta and phi_dot commands.
        :param beta_dot: command for the module's angular velocity.
        :param beta_2dot: command for the module's angular acceleration.
        :param phi_dot: command for the module wheel's angular velocity.
        :param phi_2dot: command for the module wheel's angular acceleration.
        :param delta_t: timestep over which the command will be executed.
        :param beta_e: estimate of the current beta values (angles) of the
        modules.
        :returns: beta_c, phi_dot_c (module angle and wheel angular velocity
        commands)
        """

        beta_c = beta_e + beta_dot * delta_t + 1/2 * beta_2dot * (delta_t ** 2)  # 40a

        phi_dot_c = ((phi_dot - self.b / self.r * beta_dot)
                     + (phi_2dot - self.b / self.r * beta_2dot) * delta_t)  # 40b

        beta_c = np.clip(beta_c, self.beta_bounds[0], self.beta_bounds[1])  # 41a

        phi_dot_c = np.clip(phi_dot_c, self.phi_dot_bounds[0],
                            self.phi_dot_bounds[1])  # 41b

        return beta_c, phi_dot_c
