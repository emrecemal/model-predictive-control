import numpy as np


class LinearSystem:
    def __init__(self, Ts):
        """
            Linear system with parameters, with sampling time parameter
        :param Ts: sampling time
        """
        self.m1 = 2    # kg
        self.m2 = 2    # kg
        self.k1 = 100  # N / m
        self.k2 = 200  # N / m
        self.d1 = 1    # Ns / m
        self.d2 = 5    # Ns / m
        self.Ts = Ts   # s

        # Continuous time system matrices
        self.Ac = np.array([[0, 1, 0, 0],
                       [-(self.k1 + self.k2) / self.m1, -(self.d1 + self.d2) / self.m1, self.k2 / self.m1, self.d2 / self.m1],
                       [0, 0, 0, 1],
                       [self.k2 / self.m2, self.d2 / self.m2, -self.k2 / self.m2, -self.d2 / self.m2]])
        self.Bc = np.array([[0],
                       [0],
                       [0],
                       [1 / self.m2]])
        self.Cc = np.array([[1, 0, 0, 0]])

        # State dimension
        self.n = 4

        # Number of outputs
        self.m = 1

        # Number of inputs
        self.r = 1

    def discretize_model(self):
        """
            Zero-order hold estimation
        :return:
        """
        Ad = np.eye(self.n) + \
             self.Ac * self.Ts + \
             np.linalg.matrix_power(self.Ac, 2) * self.Ts ** 2 / 2 + \
             np.linalg.matrix_power(self.Ac, 3) * self.Ts ** 3 / 6 + \
             np.linalg.matrix_power(self.Ac, 4) * self.Ts ** 4 / 24 + \
             np.linalg.matrix_power(self.Ac, 5) * self.Ts ** 5 / 120 + \
             np.linalg.matrix_power(self.Ac, 6) * self.Ts ** 6 / 720
        Bd = (Ad - np.eye(self.n)) @ np.linalg.inv(self.Ac) @ self.Bc
        Cd = self.Cc

        return Ad, Bd, Cd
