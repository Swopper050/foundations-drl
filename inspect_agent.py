import argparse

import gym
import torch


def main(args):
    env = gym.make(args.env_name)
    agent = torch.load(f"saved_agents/{args.agent_name}")

    while True:
        obs = env.reset()
        agent.episode_reset()
        done = False

        while not done:
            obs, _, done, _ = env.step(agent.act(obs))
            env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--agent-name", type=str, required=True)
    args = parser.parse_args()

    main(args)
