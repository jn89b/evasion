import numpy as np
import gymnasium as gym
import random 
from evasion.envs.Agent import Agent
from stable_baselines3 import DQN
from stable_baselines3 import PPO

# from stable_baselines.common.env_checker import check_env
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN


v_min = 0.5
v_max = 2.0

w_min = np.deg2rad(-30)
w_max = np.deg2rad(30)

goal_x = 30
goal_y = 30

min_x = 0
max_x = 200

min_y = 0
max_y = 200

#random start between 0 and 150
# start_x = random.uniform(0, 30)
# start_y = random.uniform(0, 30)
start_x = 175
start_y = 175
start_psi = random.uniform(np.deg2rad(-180), np.deg2rad(180))

start_x = 30
start_y = 50
start_psi = np.deg2rad(90)


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
TOTAL_TIMESTEPS = 500000 #100000
N_ACTIONS = 60
RENDER_FPS = 30
## 

init_states = np.array([start_x, start_y, start_psi])
evader = Agent(init_states, agent_params)

if RENDER:
    env = gym.make('evasion/MissionGym-v0', render_mode='human',
                evader=evader, render_fps=RENDER_FPS, num_discrete_actions=N_ACTIONS, 
                use_random_start=True)
else:
    env = gym.make('evasion/MissionGym-v0', render_mode=None,
                evader=evader, render_fps=RENDER_FPS, num_discrete_actions=N_ACTIONS, 
                use_random_start=True)

env._max_episode_steps = 400
    

if LOAD_MODEL:
    # model = DQN.load("dqn_missiongym")
    model = PPO.load("ppo_missiongym")
else:
    # model = DQN('MultiInputPolicy', env, 
    #             verbose=1, tensorboard_log='tensorboard_logs/',
    #             device='cuda')
    model = PPO("MultiInputPolicy", 
                env,
                learning_rate=0.0001,
                # clip_range=0.2,
                # n_epochs=10,
                # seed=42, 
                verbose=1, tensorboard_log='tensboard_logs/', 
                device='cuda')
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
    model.save("ppo_missiongym")
    print("model saved")

#view with tensorboard https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
# https://stable-baselines3.readthedocs.io/en/master/common/logger.html

obs,info = env.reset()
while True: 
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    # print(obs)
    print(rewards)
    if terminated:
        random_number = np.random.randint(0, 30)
        obs, info = env.reset(seed=40, buffer=15)
        print("reach goal", info["distance"])
        # print("obs: ", obs)
    env.render()
    

#model = DQN(MlpPolicy, env, verbose=1)
