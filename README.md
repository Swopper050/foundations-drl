# Foundations Deep Reinforcement Learning

The goal of this repository is not to create a package whose algorithms
can be used as efficiently and generally as possible. Instead, the goal is to create a package with most core Reinforcement Learning algorithms
implemented separately, solely for study purposes. Therefore, there are almost no modular components, such that the code for all algorithms can be
studied independent of each other. Furthermore comments are placed at core steps. The code base is meant as a reference when implementing the algorithms
for a specific task.

It is based on the following book:

Foundations of Deep Reinforcement Learning, Theory and Practice in Python (2020) by Laura Graesser and Wah Loon Keng

# Setup

Make sure you have `swig` installed on your computer! By default, the `classic-control` and `box2d` environments are installed with `gymnasium`.

```
git clone https://github.com/Swopper050/foundations-drl.git
cd foundations-drl/
pip install virtualenv
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

# Example run
```
python train_agent.py --env-name CartPole-v1 --algorithm reinforce --save-name cartpole_v1
python inspect_agent.py --env-name CartPole-v1 --agent-name cartpole_v1
```

# Architecture
In general there are three main components for an algorithm that are implemented:
 - ReplayMemory
 - Agent
 - Trainer

## Replay Memory
The replay memory is used to store experiences observed by the agent while it interacts in the environment. For replay memory, it matters whether or not the algorithm is on-policy or off-policy. For on-policy algorithms, all experiences need to be discraded after an update. For off-policy, this is not the case.

## Agent
This class will hold the memory, the neural network(s) and the functionality of training itself. The docs will explain what the agent will do during training, and how it behaves. It will be able to act based on an observation: `agent.act(observation)`. It must be able to perform a training step (which will be called for by the Trainer). Furthermore it most be able to store a step, i.e. gather experience. The Agent will hold the details of training.

## Trainer
The Trainer will simply implement the training loop of the algorithm. Its structure is often close to the pseudocode of the algorithms encountered in the literature. It will alternate between letting the Agent gather experience, and call the training method of the Agent. Sometimes it manages hyperparameters that are for example relevant for exploration.
