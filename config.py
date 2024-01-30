import numpy as np

v_min = 0.5
v_max = 10.0

w_min = np.deg2rad(-30)
w_max = np.deg2rad(30)

N_DISCRETE_ACTIONS = 10

v_range = np.linspace(v_min, v_max, N_DISCRETE_ACTIONS)
w_range = np.linspace(w_min, w_max, N_DISCRETE_ACTIONS)

min_x = 0
max_x = 1000

min_y = 0
max_y = 1000

min_psi = np.deg2rad(-180)
max_psi = np.deg2rad(180)

goal_x = 900
goal_y = 900

total_actions = N_DISCRETE_ACTIONS**2
