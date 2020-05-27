# Reinforcement-Learning-for-Two-Wheeled-robot

This repository contains the Model based Reinforcement Learning algorithm developed by me to work on a two-wheeled robot. 

discrete_mdp.py : Build the MDP of the robot (State and Action spaces and transition dynamics matrices). State and Action spaces are discrete.

mdp_qolo.py : The continuous version of MDP. State and Action spaces are continuous in this case.

reward.py : Contains class for designing reward functions (rbf, distance, sum)

rl_agent.py : Contains class for initiating rl_agent, which builds value-function (finite time horizon) and also generates the policy given the MDP

run_qolo.py : The main source code, where the class instances of reward, mdp and rl_agent are imported and initialized.

