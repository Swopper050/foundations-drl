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

        total_reward = 0
        while not done:
            obs, reward, done, _ = env.step(agent.act(obs))
            total_reward += reward
            env.render()
        print(total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--agent-name", type=str, required=True)
    args = parser.parse_args()

    main(args)
