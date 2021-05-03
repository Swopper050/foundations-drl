import argparse

import gym

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
    agent = trainer.train_agent(env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    args = parser.parse_args()

    main(args)
