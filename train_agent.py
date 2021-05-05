import argparse
import os

import gym
import torch

from algorithms.reinforce import ReinforceTrainer


def get_trainer(algorithm_name):
    """
    Creates an instance of a trainer for the specified algorithm name.

    :param algorithm_name: str, name of the algorithm to use
    :returns: class with type BaseTrainer
    """

    if algorithm_name == "reinforce":
        return ReinforceTrainer()

    raise ValueError("Unknown algorithm {}".format(algorithm_name))


def main(args):
    env = gym.make(args.env_name)
    trainer = get_trainer(args.algorithm)
    agent = trainer.train_agent(env=env, render=args.render)

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
