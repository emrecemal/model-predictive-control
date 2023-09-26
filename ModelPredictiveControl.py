import numpy as np
from scipy.optimize import minimize


class ModelPredictiveControl:
    def __init__(self, A, B, C, predictionHorizon, controlHorizon, initialStates, desiredControlTrajectoryTotal, weights):
        # Initialize variables
        self.A = A
        self.B = B
        self.C = C
        self.predictionHorizon = predictionHorizon
        self.controlHorizon = controlHorizon
        self.desiredTrajectoryTotal = desiredControlTrajectoryTotal

        for k, v in weights.items():
            setattr(self, k, v)

        # Dimensions of the matrices
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.r = C.shape[0]

        # Variable to track the current time step k of the controller
        # after every calculation of the control input, increment +1
        self.currentTimeStep = 0

        # Store the state vectors of the controlled state trajectory
        self.states = [initialStates]

        # Controlled inputs
        self.inputs = []
        self.inputSequences = []

        # Outputs
        self.outputs = []
        self.predictedTrajectories = []

        # Calculate the matrices to speed up the process
        self.O, self.M = self.form_matrices()

        # Calculate the weights
        self.W1, self.W2, self.W3 = self.calculate_weights()

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

    def calculate_weights(self):
        # Input costs weight
        W1 = np.zeros((self.controlHorizon * self.m, self.controlHorizon * self.m))
        for i in range(self.controlHorizon):
            if i == 0:
                W1[i * self.m: (i + 1) * self.m, i * self.m: (i + 1) * self.m] = np.eye(self.m, self.m)
            else:
                W1[i * self.m: (i + 1) * self.m, i * self.m: (i + 1) * self.m] = np.eye(self.m, self.m)
                W1[i * self.m: (i + 1) * self.m, (i - 1) * self.m: i * self.m] = -np.eye(self.m, self.m)

        W2 = np.zeros((self.controlHorizon * self.m, self.controlHorizon * self.m))
        for i in range(self.controlHorizon):
            W2[i * self.m: (i + 1) * self.m, i * self.m: (i + 1) * self.m] = self.Q[i]

        # Tracking error costs weight
        W3 = np.zeros((self.predictionHorizon * self.r, self.predictionHorizon * self.r))
        for i in range(self.predictionHorizon):
            W3[i * self.r: (i + 1) * self.r, i * self.r: (i + 1) * self.r] = self.P

        return W1, W2, W3

    def compute_control_inputs(self, method="SLSQP"):
        """
            This function computes the control inputs, applies them
            by calling the step_dynamics() function and appends that
            store the inputs, outputs, states
        :return:
        """
        desiredControlTrajectory = self.desiredTrajectoryTotal[self.currentTimeStep:self.currentTimeStep + self.predictionHorizon, :]

        if self.currentTimeStep == 0:
            u0 = np.ones((self.controlHorizon, 1))
        else:
            u0 = self.inputSequences[-1]
        res = minimize(self.get_costs, u0, method=method, args=(desiredControlTrajectory,))
        self.inputSequences.append(res.x)
        inputSequenceComputed = np.vstack(res.x)
        inputApplied = inputSequenceComputed[0]

        # Propagate Dynamics
        xkp1, yk = self.step_dynamics(np.vstack(inputApplied), self.states[self.currentTimeStep])

        # Append the lists
        self.states.append(xkp1)
        self.outputs.append(yk)
        self.inputs.append(inputApplied)

        # Increment time step
        self.currentTimeStep += 1

    def get_costs(self, u, z_des):
        # Vectorize the input value
        u = np.vstack(u)

        # Form the input cost
        Ju = u.T @ self.W1.T @ self.W2 @ self.W1 @ u

        # Tracking error costs
        self.predictedTrajectories.append(self.O @ self.states[self.currentTimeStep])
        s = z_des - self.predictedTrajectories[-1]

        # Form the tracking cost
        Jz = np.transpose(s - self.M @ u) @ self.W3 @ (s - self.M @ u)

        cost = Ju + Jz

        return cost[0][0]














