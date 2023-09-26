import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from functionMPC import simulate_system
from ModelPredictiveControl import ModelPredictiveControl
from LinearSystem import LinearSystem

# Parameters
predictionHorizon = 20
controlHorizon = 10
samplingTime = 0.05

# Define system
sys = LinearSystem(samplingTime)

# Continuous-time system
Ac, Bc, Cc = sys.Ac, sys.Bc, sys.Cc

# Discrete-time system
Ad, Bd, Cd = sys.discretize_model()


def get_step_response(Ac, Bc, Cc, Ad, Bd, Cd, timeSampleTest=500):
    uTest = np.ones((1, timeSampleTest))
    x0Test = np.zeros((Ac.shape[0], 1))
    time_points = np.arange(timeSampleTest) * samplingTime

    continuous_system = signal.StateSpace(Ac, Bc, Cc, 0)
    tc, yc = signal.step(continuous_system, X0=x0Test.reshape(Ac.shape[0],), T=time_points)

    discrete_system = signal.StateSpace(Ad, Bd, Cd, 0, dt=samplingTime)
    td, yd = signal.dstep(discrete_system, x0=x0Test.reshape(Ac.shape[0],), n=timeSampleTest)
    # my_yd, my_xd = simulate_system(Ad, Bd, Cd, uTest, x0Test)

    fig, ax = plt.subplots()
    ax.plot(tc, yc, linewidth=2, label='Step response - continuous time output')
    ax.step(td, np.squeeze(yd), label='Step response - discrete time output', where='post')
    # ax.step(td, np.squeeze(my_yd[0, :]), label='Step response - my discrete time output', where='post')
    ax.legend()
    plt.show()


def generate_reference_trajectory(trajectory_type, timeSteps=500):
    if trajectory_type == "exponential":
        timeVector = np.linspace(0, 100, timeSteps)
        desiredTrajectory = np.ones(timeSteps) - np.exp(-0.01 * timeVector)
        desiredTrajectory = desiredTrajectory.reshape((timeSteps, 1))
    elif trajectory_type == "pulse":
        desiredTrajectory = np.zeros(shape=(timeSteps, 1))
        for i in range(timeSteps):
            if i % 100 == 0 and (i / 100) % 2 == 0:
                desiredTrajectory[i:i + 100, :] = np.ones((100, 1))
    elif trajectory_type == "step":
        desiredTrajectory = np.ones(shape=(timeSteps, 1))
    else:
        raise ValueError("Trajectory type should be: exponential, pulse or step!")
    return desiredTrajectory

# Simulate MPC algorithm
# Initial state
x0 = np.zeros((Ad.shape[0], 1))
timeSteps = 500

# Weights
weights = {"P": 10, "Q": 1e-4 * np.ones(controlHorizon)}
weights["Q"][0] = 1.1e-9

desiredTrajectory = generate_reference_trajectory("pulse", timeSteps)

# Controller
mpc = ModelPredictiveControl(Ad, Bd, Cd, predictionHorizon, controlHorizon, x0, desiredTrajectory, weights)

# Simulate
for i in range(timeSteps - predictionHorizon):
    mpc.compute_control_inputs()


# Extract the state estimates in order to plot the results
desiredTrajectoryList = []
controlledTrajectoryList = []
controlInputList = []
for j in np.arange(timeSteps - predictionHorizon):
    controlledTrajectoryList.append(mpc.outputs[j][0, 0])
    desiredTrajectoryList.append(desiredTrajectory[j, 0])
    controlInputList.append(mpc.inputs[j][0])

print(mpc.predictedTrajectories)

# plot the results
fig, ax = plt.subplots()
ax.plot(controlledTrajectoryList, linewidth=4, label='Controlled trajectory')
ax.plot(desiredTrajectoryList, 'r', linewidth=2, label='Desired trajectory')

fig, ax = plt.subplots()
ax.plot(controlInputList, linewidth=4, label='Computed inputs')
plt.show()
