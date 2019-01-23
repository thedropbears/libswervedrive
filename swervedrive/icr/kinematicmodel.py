from enum import Enum
import math
import numpy as np

from swervedrive.icr.estimator import shortest_distance


def cartesian_to_lambda(x, y):
    return np.reshape(1 / np.linalg.norm([x, y, 1]) * np.array([x, y, 1]), (3, 1))


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

        assert len(alpha.shape) == 2 and alpha.shape[1] == 1, alpha
        assert len(l.shape) == 2 and l.shape[1] == 1, l
        assert len(b.shape) == 2 and b.shape[1] == 1, b
        assert len(r.shape) == 2 and r.shape[1] == 1, r

        self.alpha = alpha
        self.b = b
        self.r = r
        self.l = l

        n = len(alpha)
        self.n_modules = n
        self.k_beta = k_beta
        self.r = np.reshape(r, (n, 1))

        self.a = np.concatenate([np.cos(alpha).T, np.sin(alpha).T, [[0.0] * n]])
        self.a_orth = np.concatenate([-np.sin(alpha).T, np.cos(alpha).T, [[0.0] * n]])
        self.s = np.concatenate(
            [(l * np.cos(alpha)).T, (l * np.sin(alpha)).T, [[1.0] * n]]
        )
        self.b_vector = np.concatenate([[[0.0] * n], [[0.0] * n], self.b.T])
        self.l_vector = np.concatenate([[[0.0] * n], [[0.0] * n], self.l.T])
        self.state = KinematicModel.State.STOPPING

        self.xi = np.array([[0.0]] * 3)  # Odometry

        singularities_cartesian = np.concatenate(
            [(l * np.cos(alpha)).T, (l * np.sin(alpha)).T]
        ).T
        singularities_lmda = np.array(
            [
                cartesian_to_lambda(s[0], s[1])
                for s in singularities_cartesian
            ]
        )
        self.singularities = np.concatenate([singularities_lmda, -singularities_lmda])

    def compute_chassis_motion(
        self,
        lmda_d: np.ndarray,
        lmda_e: np.ndarray,
        mu_d: float,
        mu_e: float,
        k_b: float,
        phi_dot_bounds: float,
        k_lmda: float,
        k_mu: float,
    ):
        """
        Compute the path to the desired state and implement control laws
        required to produce the motion.
        :param lmda_d: The desired ICR.
        :param lmda_e: Estimate of the current ICR.
        :param mu_d: Desired motion about the ICR.
        :param mu_e: Estimate of the current motion about the ICR.
        :param k_b: Backtracking constant.
        :param phi_dot_bounds: Min/max allowable value for rotation rate of
            module wheels, in rad/s
        :param k_lmda: Proportional gain for the movement of lmda. Must be >=1
        :param k_mu: Proportional gain for movement of mu. Must be >=1
        :returns: (derivative of lmda, 2nd derivative of lmda, derivative of mu)
        """

        assert lmda_d.shape == (3,1), lmda_d
        assert lmda_e.shape == (3,1), lmda_e

        # Because +lmda and -lmda are the same, we should choose the closest one
        if lmda_d.T.dot(lmda_e) < 0:
            lmda_d = -lmda_d
            mu_d = -mu_d

        # bound mu based on the ph_dot constraits
        mu_min = max(self.compute_mu(lmda_d, phi_dot_bounds[0]))
        mu_max = min(self.compute_mu(lmda_d, phi_dot_bounds[1]))
        mu_d = max(min(mu_d, mu_max), mu_min)

        # TODO: figure out what the tolerance should be
        on_singularity = any(
            lmda_d.T.dot(s) >= 0.99 for s in self.singularities
        )
        if on_singularity:
            lmda_d = lmda_e

        dlmda = k_b * k_lmda * (lmda_d - (lmda_e.T.dot(lmda_d)) * lmda_e)

        d2lmda = k_b ** 2 * k_lmda ** 2 * ((lmda_e.T.dot(lmda_d)) * lmda_d - lmda_e)

        dmu = k_b * k_mu * (mu_d - mu_e)

        return dlmda, d2lmda, dmu

    def compute_mu(self, lmda: np.ndarray, phi_dot: float):
        """
        Compute mu given lmda and phi_dot (equation 25 of control paper).
        """

        assert lmda.shape == (3,1), lmda

        lmda = np.reshape(lmda, (-1, 1))
        _, s_perp_2 = self.s_perp(lmda)
        # TODO: change this line
        f_lmda = (self.r / (s_perp_2 - self.b_vector).T.dot(lmda)).reshape(-1)
        return f_lmda * phi_dot

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
        assert lmda.shape == (3,1), lmda
        assert lmda_dot.shape == (3,1), lmda_dot
        assert lmda_2dot.shape == (3,1), lmda_2dot

        if self.state == KinematicModel.State.STOPPING:
            if abs(mu) < 1e-3:
                # We are stopped, so we can reconfigure
                self.state = KinematicModel.State.RECONFIGURING

        s1_lmda, s2_lmda = self.s_perp(lmda)

        denom = s2_lmda.T.dot(lmda)
        # Any zeros in denom represent an ICR on a wheel axis
        # Set the corresponding beta_prime and beta_2prime to 0
        denom[denom == 0] = 1e20
        beta_prime = -(s1_lmda.T.dot(lmda_dot)) / denom

        assert beta_prime.shape == (self.n_modules, 1)

        beta_2prime = (
            -(
                2 * np.multiply(beta_prime, s2_lmda.T.dot(lmda_dot))
                + s1_lmda.T.dot(lmda_2dot)
            )
            / denom
        )

        assert beta_2prime.shape == (self.n_modules, 1)

        phi_dot = np.divide(
            (s2_lmda - self.b_vector).T.dot(lmda) * mu
            - np.multiply(self.b, beta_prime),
            self.r,
        )

        assert phi_dot.shape == (self.n_modules, 1)

        phi_dot_prime = np.divide(
            (
                (s2_lmda - self.b_vector).T.dot(
                    np.multiply(lmda_dot, mu) + np.multiply(lmda, mu_dot)
                )
                - np.multiply(self.b, beta_2prime)
            ),
            self.r,
        )

        assert phi_dot_prime.shape == (self.n_modules, 1)

        return (
            np.reshape(beta_prime, (-1, 1)),
            np.reshape(beta_2prime, (-1, 1)),
            np.reshape(phi_dot, (-1, 1)),
            np.reshape(phi_dot_prime, (-1, 1)),
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
        assert len(beta_d.shape) == 2 and beta_d.shape[1] == 1, beta_d
        assert len(beta_e.shape) == 2 and beta_e.shape[1] == 1, beta_e

        error = shortest_distance(beta_d, beta_e)
        dbeta = self.k_beta * error
        if np.linalg.norm(error) < 1/180*math.pi * self.n_modules:  # degrees per module
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

        assert lmda_e.shape == (3,1), lmda_e

        xi_dot = (mu_e * np.array([lmda_e[1], -lmda_e[0], lmda_e[2]])).reshape(-1,1)  # Eq (2)
        theta = self.xi[2, 0]
        m3 = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        self.xi += np.matmul(m3, xi_dot) * delta_t  # Eq (24)
        return self.xi

    def estimate_mu(self, phi_dot: np.ndarray, lmda_e):
        """
        Find the rotational position of the robot about the ICR.
        :param phi_dot: array of angular velocities of the wheels.
        :param lmda_e: the estimate of the ICR in h-space.
        :returns: the estimate of mu (float).
        """

        assert len(phi_dot.shape) == 2 and phi_dot.shape[1] == 1, phi_dot
        assert lmda_e.shape == (3,1), lmda_e

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

        assert lmda.shape == (3,1), lmda

        s = np.dot(self.a_orth.T, lmda)
        c = np.dot((self.a - self.l_vector).T, lmda)
        s1_lmda = (
            np.multiply(s, (self.a - self.l_vector).T) - np.multiply(c, self.a_orth.T)
        ).T
        s2_lmda = (
            np.multiply(c, (self.a - self.l_vector).T) + np.multiply(s, self.a_orth.T)
        ).T

        assert s1_lmda.shape == (3, self.n_modules)
        assert s2_lmda.shape == (3, self.n_modules)

        return s1_lmda, s2_lmda
