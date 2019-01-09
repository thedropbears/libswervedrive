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

        self.a = np.array([np.cos(alpha), np.sin(alpha), [0.0] * n])
        self.a_orth = np.array([-np.sin(alpha), np.cos(alpha), [0.0] * n])
        self.s = np.array(
            [np.multiply(l, np.cos(alpha)), np.multiply(l, np.sin(alpha)), [1.0] * n]
        )
        self.b = np.reshape(b, (n, 1))
        self.l = np.reshape(l, (n, 1))
        self.b_vector = np.array([[0.0] * n, [0.0] * n, b])
        self.l_vector = np.array([[0.0] * n, [0.0] * n, l])
        self.state = KinematicModel.State.STOPPING

        self.xi = np.array([[0.0] * 3])  # Odometry

    def compute_chassis_motion(self, lmda_d: np.ndarray, lmda_e: np.ndarray,
                               mu_d: float, mu_e: float, k_b: float,
                               k_lmda: float, k_mu: float):
        """
        Compute the path to the desired state and implement control laws
        required to produce the motion.
        :param lmda_d: The desired ICR.
        :param lmda_e: Estimate of the current ICR.
        :param mu_d: Desired motion about the ICR.
        :param mu_e: Estimate of the current motion about the ICR.
        :param k_b: Backtracking constant.
        :param k_lmda: Proportional gain for the movement of lmda. Must be >=1
        :param k_mu: Proportional gain for movement of mu. Must be >=1
        :returns: (derivative of lmda, 2nd derivative of lmda, derivative of mu)
        """

        dlmda = k_b * k_lmda * (lmda_d - (lmda_e.dot(lmda_d)) * lmda_e)

        d2lmda = k_b ** 2 * k_lmda ** 2 * ((lmda_e.dot(lmda_d)) * lmda_d - lmda_e)

        dmu = k_b * k_mu * (mu_d-mu_e)

        return dlmda, d2lmda, dmu

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

        lmda = np.reshape(lmda, (-1, 1))
        lmda_dot = np.reshape(lmda_dot, (-1, 1))
        lmda_2dot = np.reshape(lmda_2dot, (-1, 1))
        s1_lmda, s2_lmda = self.s_perp(lmda)

        denom = s2_lmda.T.dot(lmda)
        # Any zeros in denom represent an ICR on a wheel axis
        # Set the corresponding beta_prime and beta_2prime to 0
        denom[denom == 0] = 1e20
        beta_prime = -(s1_lmda.T.dot(lmda_dot)) / denom

        beta_2prime = (
            -(
                2 * np.multiply(beta_prime, s2_lmda.T.dot(lmda_dot))
                + s1_lmda.T.dot(lmda_2dot)
            )
            / denom
        )

        phi_dot = np.divide(
            (s2_lmda - self.b_vector).T.dot(lmda) * mu
            - np.multiply(self.b, beta_prime),
            self.r,
        )

        phi_dot_prime = np.divide(
            (
                (s2_lmda - self.b_vector).T.dot(
                    np.multiply(lmda_dot, mu) + np.multiply(lmda, mu_dot)
                )
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

    def compute_odometry(self, lmda_e: np.ndarray, mu_e: float, delta_t: float):
        """
        Update our estimate of epsilon (twist position) based on the new ICR
        estimate.
        :param lmda_e: the estimate of the ICR in h-space.
        :param mu_e: estimate of the position of the robot about the ICR.
        :param delta_t: time since the odometry was last updated.
        """
        xi_dot = mu_e * np.array([[lmda_e[1]], [-lmda_e[0]], [lmda_e[2]]])  # Eq (2)
        theta = self.xi[0, 2]
        m3 = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        self.xi += np.matmul(m3, xi_dot).T * delta_t  # Eq (24)
        return self.xi

    def estimate_mu(self, phi_dot: np.ndarray, lmda_e):
        """
        Find the rotational position of the robot about the ICR.
        :param phi_dot: array of angular velocities of the wheels.
        :param lmda_e: the estimate of the ICR in h-space.
        :returns: the estimate of mu (float).
        """
        # this requires solving equation (22) from the control paper, i think
        # we may need to look into whether this is valid for a system with no
        # wheel coupling
        lmda_e = np.reshape(lmda_e, (-1, 1))
        phi_dot = np.reshape(phi_dot, (-1, 1))
        s1_lmda, s2_lmda = self.s_perp(lmda_e)
        C = np.multiply(1.0 / s2_lmda.T.dot(lmda_e), s1_lmda.T)
        D = (s2_lmda - self.b_vector).T.dot(lmda_e) / self.r
        # Build the matrix
        K_lmda = np.block([[lmda_e.T, 0.0], [self.b / self.r * C, D]])
        phi_dot_augmented = np.block([[0], [phi_dot]])
        state = np.linalg.lstsq(K_lmda, phi_dot_augmented, rcond=None)[0]
        mu = state[-1, 0]
        return mu

    def s_perp(self, lmda: np.ndarray):
        s = np.dot(self.a_orth.T, lmda)
        c = np.dot((self.a - self.l_vector).T, lmda)
        s1_lmda = (
            np.multiply(s, (self.a - self.l_vector).T) - np.multiply(c, self.a_orth.T)
        ).T
        s2_lmda = (
            np.multiply(c, (self.a - self.l_vector).T) + np.multiply(s, self.a_orth.T)
        ).T
        return s1_lmda, s2_lmda
