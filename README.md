This repository contains the solution for the first project of the Udacity Deep Reinforcement Learning Nanodegree.

# The environment

The project environment is a modified version of the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). The agent is trained to navigate and collect bananas in a large, square world.

## Environment description

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* **0** - move forward.
* **1** - move backward.
* **2** - turn left.
* **3** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

# How to

Install the dependancies first

* **Option 1: pipenv (recommended).** Initialize a pipenv environment by running `pipenv --three install` inside the root directory of the repo. [Pipenv](http://docs.pipenv.org/) will automatically locate the [Pipfiles](https://github.com/pypa/pipfile), create a new virtual environment and install the necessary packages.

* **Option 2: pip.** Install the needed dependencies by running `pip install -r requirements.txt` 

A solution of the environment can be obtained by running the `DQN_Navigation.ipynb` notebook.

## Repository structure

*  `DQN_Navigation.ipynb` notebook contains a solution of the environment
*  `DQN_Navigation_Hyperparameters.ipynb` notebook contains an exploration of hyperparameters effects
*  `Report.md` contains a description of the implementation.
*  `trainer.py` contains the code for running the training of the agent over a given number of episodes
*  `agent.py` contains the implementation of the deep Q network agent
*  `model.py` contains the definition of the Q neural network(s)
*  `checkpoint.pth` contains the model weights of the network of the environment solution

All the listed `.py` files are highely parametrizable in all the important parameters.





