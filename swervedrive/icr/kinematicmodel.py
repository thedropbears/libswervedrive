from enum import Enum
import numpy as np


class KinematicModel:
    class State(Enum):
        STOPPING = 1
        RECONFIGURING = 2
        RUNNING = 3

    def __init__(
        self,
        alpha: np.ndarray,
        l: np.ndarray,
        b: np.ndarray,
        r: np.ndarray,
        k_beta: float,
    ):
        """
        Initialize the KinematicModel object. The order in the following arrays
        must be preserved throughout all arguments passed to this object.
        :param alpha: array containing the angle to each of the modules,
        measured counter clockwise from the x-axis (rad).
        :param l: distance to the axis of rotation of each module from
        the origin of the chassis frame (m).
        :param b: distance from the axis of rotation of each module to
        it's contact with the ground (m).
        :param r: radii of wheels (m).
        :param k_beta: the gain for wheel reconfiguration.
        """
        self.alpha = alpha
        n = len(alpha)
        self.n_modules = n
        self.k_beta = k_beta
        self.r = np.reshape(r, (n, 1))

        self.a = np.array([np.cos(alpha), np.sin(alpha), [0] * n])
        self.a_orth = np.array([-np.sin(alpha), np.cos(alpha), [0] * n])
        self.s = np.array(
            [np.multiply(l, np.cos(alpha)), np.multiply(l, np.sin(alpha)), [1] * n]
        )
        self.b = np.reshape(b, (n, 1))
        self.l = np.reshape(l, (n, 1))
        self.b_vector = np.array([[0] * n, [0] * n, b])
        self.l_vector = np.array([[0] * n, [0] * n, l])
        self.state = KinematicModel.State.STOPPING

    def compute_actuators_motion(
        self,
        lmda: np.ndarray,
        lmda_dot: np.ndarray,
        lmda_2dot: np.ndarray,
        mu: float,
        mu_dot: float,
    ):
        """
        Compute the motion of the actuators based on the chassis motion commands.
        :param lmda: lambda, as a column vector.
        :param lmda_dot: first derivative of lambda, as a column vector.
        :param lmda_2dot: second derivative of lambda, as a column vector.
        :param mu: mu (rad/s).
        :param mu_dot: derivative of mu (rad/s^2).
        :returns: first and second derivatives of the actuators' positions, as
        arrays: (beta_prime, beta_2prime, phi_dot, phi_dot_prime) (phi_dot is a already a time
        derivative as the relevant constraints are applied in the path planner).
        """

        if self.state == KinematicModel.State.STOPPING:
            if abs(mu) < 1e-3:
                # We are stopped, so we can reconfigure
                self.state = KinematicModel.State.RECONFIGURING

        lmda = np.reshape(lmda, (len(lmda), 1))
        lmda_dot = np.reshape(lmda_dot, (len(lmda_dot), 1))
        lmda_2dot = np.reshape(lmda_2dot, (len(lmda_2dot), 1))
        s1_lmda = np.multiply(np.sin(lmda), (self.a - self.l_vector)) - np.multiply(
            np.cos(lmda), self.a_orth
        )
        s2_lmda = np.multiply(np.cos(lmda), (self.a - self.l_vector)) + np.multiply(
            np.sin(lmda), self.a_orth
        )

        denom = s2_lmda.transpose().dot(lmda)
        # Any zeros in denom represent an ICR on a wheel axis
        # Set the corresponding beta_prime and beta_2prime to 0
        denom[denom == 0] = 1e20
        beta_prime = -(s1_lmda.transpose().dot(lmda_dot)) / denom

        beta_2prime = (
            -(
                2 * np.multiply(beta_prime, s2_lmda.transpose().dot(lmda_dot))
                + s1_lmda.transpose().dot(lmda_2dot)
            )
            / denom
        )

        phi_dot = np.divide(
            (s2_lmda - self.b_vector).transpose().dot(lmda) * mu
            - np.multiply(self.b, beta_prime),
            self.r,
        )

        phi_dot_prime = np.divide(
            (
                (s2_lmda - self.b_vector)
                .transpose()
                .dot(np.multiply(lmda_dot, mu) + np.multiply(lmda, mu_dot))
                - np.multiply(self.b, beta_2prime)
            ),
            self.r,
        )

        return (
            np.reshape(beta_prime, (self.n_modules,)),
            np.reshape(beta_2prime, (self.n_modules,)),
            np.reshape(phi_dot, (self.n_modules,)),
            np.reshape(phi_dot_prime, (self.n_modules,)),
        )

    def reconfigure_wheels(self, beta_d: np.ndarray, beta_e: np.ndarray):
        """
        Perform a wheel reconfiguration to a desired state.
        This method must be called *instead* of compute actuators motion when
        the ICR is on a structural singularity, as in that case there are no
        valid wheel configurations.
        :beta_d: array of desired beta values
        :beta_e: array of measured beta values.
        :returns: Array of dbeta values.
        """
        error = beta_d - beta_e
        dbeta = self.k_beta * error
        if np.isclose(dbeta, 0, atol=1e-2).all():
            self.state = KinematicModel.State.RUNNING
        return dbeta
