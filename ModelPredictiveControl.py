import numpy as np


class ModelPredictiveControl:
    def __init__(self, A, B, C, predictionHorizon, controlHorizon):
        # Initialize variables
        self.A = A
        self.B = B
        self.C = C
        self.predictionHorizon = predictionHorizon
        self.controlHorizon = controlHorizon

        # Dimensions of the matrices
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.r = C.shape[0]

    def form_matrices(self):
        # Output state matrix, O
        O = np.zeros(shape=(self.predictionHorizon * self.r, self.n))
        for i in range(self.predictionHorizon):
            if i == 0:
                powA = self.A
            else:
                powA = powA @ self.A
            O[i * self.r:(i + 1) * self.r, :] = self.C @ powA

        # Controlled input matrix, M
        M = np.zeros(shape=(self.predictionHorizon * self.r, self.controlHorizon * self.m))
        for i in range(self.predictionHorizon):
            # until the control horizon
            if i < self.controlHorizon:
                for j in range(i + 1):
                    if j == 0:
                        powA = np.eye(self.n)
                    else:
                        powA = powA @ self.A
                    M[i * self.r:(i + 1) * self.r, (i - j) * self.m : (i - j + 1) * self.m] = self.C @ (powA @ self.B)
            # from control horizon to the prediction horizon
            else:
                for j in range(self.controlHorizon):
                    if j == 0:
                        sumLast = np.zeros(shape=(self.n, self.n))
                        for k in range(i - self.controlHorizon + 2):
                            if k == 0:
                                powA = np.eye(self.n)
                            else:
                                powA = powA @ self.A
                            sumLast += powA
                        M[i * self.r:(i + 1) * self.r, (self.controlHorizon - 1) * self.m:self.controlHorizon * self.m] = self.C @ (sumLast @ self.B)
                    else:
                        powA = powA @ self.A
                        M[i * self.r:(i + 1) * self.r, (self.controlHorizon - 1 - j) * self.m:(self.controlHorizon - j) * self.m] = self.C @ (powA @ self.B)

        return O, M

    def step_dynamics(self, u, x):
        """
            This function propagates the dynamics
            x_{k + 1} = A @ x_{k} + B @ u_{k}
        :param u: control input
        :param x: state
        :return: next states and the output
        """
        xkp1 = self.A @ x + self.B @ u
        yk = self.C @ x
        return xkp1, yk






