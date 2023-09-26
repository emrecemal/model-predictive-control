import numpy as np


def simulate_system(A, B, C, u, x0):
    simTime = u.shape[1]
    n = A.shape[0]
    r = C.shape[0]
    X = np.zeros(shape=(n, simTime + 1))
    Y = np.zeros(shape=(r, simTime))

    for i in range(0, simTime):
        if i == 0:
            X[:, [i]] = x0
            Y[:, [i]] = C @ x0
            X[:, [i + 1]] = A @ x0 + B @ u[:, [i]]
        else:
            Y[:, [i]] = C @ X[:, [i]]
            X[:, [i + 1]] = A @ X[:, [i]] + B @ u[:, [i]]

    return Y, X