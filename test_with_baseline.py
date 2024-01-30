import numpy as np
import gymnasium as gym
from evasion.envs.Agent import Agent
from stable_baselines3 import DQN
from stable_baselines3 import PPO

# from stable_baselines.common.env_checker import check_env
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

start_x = 10
start_y = 10
start_psi = 0

v_min = 0.5
v_max = 5.0
w_min = np.deg2rad(-30)
w_max = np.deg2rad(30)
goal_x = 70
goal_y = 70
min_x = 0
max_x = 100
min_y = 0
max_y = 100
min_psi = np.deg2rad(-180)
max_psi = np.deg2rad(180)

agent_params = {
    "v_min": v_min,
    "v_max": v_max,
    "w_min": w_min,
    "w_max": w_max,
    "goal_x": goal_x,
    "goal_y": goal_y,
    "min_x": min_x,
    "max_x": max_x,
    "max_y": max_y,
    "min_y": min_y,
    "min_psi": min_psi,
    "max_psi": max_psi,
}

RENDER = True
LOAD_MODEL = True
TOTAL_TIMESTEPS = 100000/2 #100000
N_ACTIONS = 50

## 

init_states = np.array([start_x, start_y, start_psi])
evader = Agent(init_states, agent_params)

if RENDER:
    env = gym.make('evasion/MissionGym-v0', render_mode='human',
                evader=evader, render_fps=60, num_discrete_actions=N_ACTIONS)
else:
    env = gym.make('evasion/MissionGym-v0', render_mode=None,
                evader=evader, render_fps=60, num_discrete_actions=N_ACTIONS)

env._max_episode_steps = 500
    

if LOAD_MODEL:
    # model = DQN.load("dqn_missiongym")
    model = PPO.load("ppo_missiongym")
else:
    # model = DQN('MultiInputPolicy', env, 
    #             learning_rate=0.01,
    #             verbose=1, tensorboard_log='tensorboard_logs/',
    #             device='cuda')
    model = PPO("MultiInputPolicy", 
                env, 
                verbose=1, tensorboard_log='tensboard_logs/', 
                device='cuda')
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
    model.save("ppo_missiongym")
    print("model saved")

#view with tensorboard https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
# https://stable-baselines3.readthedocs.io/en/master/common/logger.html

obs,info = env.reset()
while True: 
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        print("reach goal", terminated)
        # print("obs: ", obs)
    env.render()
    

#model = DQN(MlpPolicy, env, verbose=1)
