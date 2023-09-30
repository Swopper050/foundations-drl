import argparse
import os

import gymnasium as gym
import torch


def main(args):
    env = gym.make(args.env_name, render_mode="human")
    agent = torch.load(f"saved_agents/{args.agent_name}")

    while True:
        obs, _ = env.reset()
        agent.episode_reset()
        done = False

        total_reward = 0
        while not done:
            obs, reward, done, _, _ = env.step(agent.act(obs))
            total_reward += reward
            env.render()
        print(total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        type=str,
        required=True,
        help="Gym environment to test the agent on",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        required=True,
        choices=os.listdir("saved_agents"),
        help="Name of the agent to load",
    )
    args = parser.parse_args()

    main(args)
