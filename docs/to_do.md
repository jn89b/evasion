- Build a toy environment for nonholonomic vehicle using Open AI gym
- Determine what action space it should be refer to this as inspiration: https://highway-env.farama.org/actions/index.html:
    - continous: control steering and throttle
    - discrete meta: go left, idle right, etc.

- Given an agent in an global known environment:
- Objective is to reach a goal location while avoiding threats
- Outputs:
    - Turn angle 
    - Position

- Step command:
    - Make agent move
    - Make threats move
    - observations

- Teddy's notes:
    - Check out stable baseline ap:
        - https://stable-baselines3.readthedocs.io/en/master/
    - Read results:
        - https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2
        - https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
        
    - Selecting algos:
        - https://intellabs.github.io/coach/selecting_an_algorithm.html
