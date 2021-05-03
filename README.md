# Foundations Deep Reinforcement Learning

Implementing the base algorithms of Deep Reinforcement Learning in Python. The goal of this repository is not to create a package whose algorithms
can be used as efficiently and generally as possible. Instead, the goal is to create a package with most core Reinforcement Learning algorithms
implemented separately, solely for study purposes. Therefore, there are almost no modular components, such that the code for all algorithms can be
studied independent of each other, and to document them all well.

Based on the following book:

Foundations of Deep Reinforcement Learning, Theory and Practice in Python (2020) by Laura Graesser and Wah Loon Keng

# Setup
```
git clone https://github.com/Swopper050/foundations-drl.git
cd foundations-drl/
pip install virtualenv
python -m venv .env
source ./bin/activate
pip install -r requirements.txt
```

# Example run
```
python train_agent --env-name CartPole-v0 --algorithm reinforce
```
