clc; clear all; close all;

% System parameters
m1 = 2;    % kg
m2 = 2;    % kg
k1 = 100;  % N / m
k2 = 200;  % N / m
d1 = 1;    % Ns / m
d2 = 5;    % Ns / m

% Continious System Matrices
Ac = [              0,               1,        0,        0;
      -(k1 + k2) / m1, -(d1 + d2) / m1,  k2 / m1,  d2 / m1;
                    0,               0,        0,        1;
              k2 / m2,         d2 / m2, -k2 / m2, -d2 / m2];
Bc = [    0; 
          0; 
          0; 
    1 / m2];
Cc = [1, 0, 0, 0];
Dc = 0;

% Create state space plant model
continuousSystem = ss(Ac, Bc, Cc, Dc);

% Set names
continuousSystem.InputName = {'F'};     % force
continuousSystem.OutputName = {'x_1'};  % position of the first mass 
continuousSystem.StateName = {'x_1', 'v_1', 'x_2', 'v_2'};  % state names

% Set units
continuousSystem.InputUnit = {'N'};
continuousSystem.OutputUnit = {'m'};
continuousSystem.StateUnit = {'m', 'm/s', 'm', 'm/s'};

% Set MPC signals
continuousSystem = setmpcsignals(continuousSystem, 'MV', 1, 'MO', 1);

% Create controller

% Controller Design Parameters
Ts = 0.05;  % s
Np = 20;
Nc = 10;

% Create the model predictive controller
mpcobj = mpc(continuousSystem, Ts, Np, Nc);

% Set constraints for the controller manipulated variable
mpcobj.W.OV = 10;
mpcobj.W.MVRate = [1.1e-9; 1e-4];

% Review
review(mpcobj)
sensitivity(mpcobj)

% Generate Pulse Reference
T = 25;  % s
timesteps = T / Ts;
r = zeros(timesteps, 1);
for i = 1:timesteps
    if mod(i, 100) == 1 && mod(i / int16(100), 2) == 0
        r(i:i+99, :) = ones(100, 1);
    end
end

% Simulate
sim(mpcobj, timesteps, r)