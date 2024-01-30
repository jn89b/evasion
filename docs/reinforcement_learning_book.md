## General Idea
- $A$ - Agent 
- $R$ - Reward
- $E$ - Envioronment 
- $A$ - Action
- $S$ - State

An Agent, $A$, wants to maximize $R$ in $E$ based on the Actions it does $A$ at each state $S$. This is all formulated as trying to solve an optimal control problem in an incomplete Markov decision process(es). 

# Chapter 2 

## I. Tabular Solution Methods
- Core idea of RL in most simple form. Action states and rewards are small enough to be represented in arrays or tables. 
- Ways to sovle:
    - RL problems with single state: Bandit problem
    - RL problems with finite Markov Decision Processes (MDP):
        - Dynamic Programming - mathematical formulations, require model and suffer from curse of dimensionality. The more the states the harder the formulation
        - Monte Carlo - don't require model but not well suited for step-by-step incremental computation
        - Temporal difference methods - don't require model, fully incremental but hard to analyze 

### Multi-Armed Bandits
- Given $k$ options over some time interval
- Each choice you make gives you a reward value
- Goal is to maximize the total reward value over the the entire time interval/trajectory
- Based on slot machine, where you have $k$ levers or 


$q^*(a) = \mathbb{E}[R_t \mid A_t = a]$

We denote the action selected on time step $t$ as $A_t$, and the corresponding reward as Rt. The value then of an arbitrary action a, denoted $q^*(a)$, is the expected reward given that $a$ is selected
- Now you can either be $greedy$ and try to find the action that will maximize your payoff 
- Or you can not be greedy and look at other actions, this is known as $exploring$.


### 2.2 Action-Value Methods
- Motivation: How do we estimate the values of actions? Especially if we don't know what the actual value is? , we can try to estimate them. 


$Q_t(a) \coloneqq \frac{\sum\limits_{i=1}^{t-1} R_i \cdot \mathbb{I}_{A_i=a}}{\sum\limits_{i=1}^{t-1} \mathbb{I}_{A_i=a}}$

## Why Deep Q Learning
- Q-learning is cool but when your actions and states explode up can't tabulate the value anymore 
- Structure is like this:
    - Inptus are your states
    - Outputs are the optimal states
- Loss/Cost function is based on the Bellman equation 
- We want to minimize this loss utilizing your traditional NN scheme with backprogration