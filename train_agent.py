import argparse
import os

import gym
import torch

from algorithms.dqn import DQNTrainer
from algorithms.reinforce import ReinforceTrainer
from algorithms.sarsa import SarsaTrainer
from algorithms.vanilla_dqn import VanillaDQNTrainer


def get_trainer(algorithm_name):
    """
    Creates an instance of a trainer for the specified algorithm name.

    :param algorithm_name: str, name of the algorithm to use
    :returns: class with type BaseTrainer
    """

    if algorithm_name == "reinforce":
        return ReinforceTrainer()
    elif algorithm_name == "sarsa":
        return SarsaTrainer()
    elif algorithm_name == "vanilla_dqn":
        return VanillaDQNTrainer()
    elif algorithm_name == "dqn":
        return DQNTrainer()

    raise ValueError("Unknown algorithm {}".format(algorithm_name))


def main(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    trainer = get_trainer(args.algorithm)
    agent = trainer.train_agent(env=env, test_env=test_env, render=args.render)

    if not os.path.exists("saved_agents"):
        os.makedirs("saved_agents")
    torch.save(agent, f"saved_agents/{args.save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--save-name", type=str, default="test_agent")
    args = parser.parse_args()

    main(args)
