import numpy as np
from collections import defaultdict
import argparse
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import os

class SARSA:
    def __init__(self,num_state:int,num_action:int,low_state:np.array,high_state:np.array, num_discrete:int=50):
        self.table = defaultdict(float)
        self.low_state = low_state
        self.high_state = high_state
        self.num_action = num_action
        self.num_discrete = num_discrete

    def get(self, state:np.array, action:int):
        index = self.get_index(state,action)
        return self.table[index]

    def set(self, state:np.array, action:int, value:float):
        index = self.get_index(state,action)
        self.table[index] = value

    def get_index(self,state:np.array,action:int):
        index_list=[]
        state = state[2:]

        for i in range(len(state)):
            index = (state[i]-self.low_state[i])//((self.high_state[i]-self.low_state[i])/self.num_discrete)
            index = int(index)
            index_list.append(index)

        index_list.append(action)
        return tuple(index_list)

    def get_action(self,state:np.array):
        max_value = -float('inf')
        action=None

        for i in range(self.num_action):
            index = self.get_index(state,i)
            value = self.table[index]

            if value > max_value:
                max_value = value
                action = i

        return action

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

def train(args, env, agent:SARSA):
    epsilon = 0.2
    max_episode_reward = -float('inf')
    episode_reward = 0
    episode_length = 0
    log = defaultdict(list)

    state,_ = env.reset(seed=args.seed)

    for step in range(args.max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, terminated,truncated,_ = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1

        if np.random.rand() < epsilon:
            new_action = env.action_space.sample()
        else:
            new_action = agent.get_action(next_state)

        target = reward +args.gamma * agent.get(next_state, new_action)
        td_delta = agent.get(state, action) - target

        value = agent.get(state, action) - args.lr * td_delta
        agent.set(state, action, value)

        state = next_state

        if done is True:
            log['episode_reward'].append(episode_reward)
            log['episode_length'].append(episode_length)

            max_episode_reward = max(log["episode_reward"])
            print(f"step={step}, reward={episode_reward}, length={episode_length}, max_reward={max_episode_reward}, epsilon={epsilon:.2f}")

            episode_reward = 0
            episode_length = 0
            state,_ = env.reset()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.plot(np.cumsum(log["episode_length"]), log["episode_reward"])
    plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
    plt.close()

def test(args, env, agent:SARSA)->None:

    #naive_env=env
    env = gym.wrappers.RecordVideo(env,video_folder=args.output_dir)

    state, _ = env.reset()
    done=False
    score=0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated,truncated,_ = env.step(action)
        done = terminated or truncated
        state = next_state
        score += reward

    print("score:",score)
    env.close()

    #env = naive_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment name.")
    parser.add_argument("--num_state", default=4, type=int, help="Dimension of observation.")
    parser.add_argument("--num_action", default=2, type=int, help="Number of actions.")

    parser.add_argument("--gamma", default=0.95, type=float, help="Discount coefficient.")
    parser.add_argument("--max_steps", default=200_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--lr", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")

    args = parser.parse_args()

    env = gym.make(args.env, max_episode_steps=200,render_mode='rgb_array')

    set_seed(args)

    low_state = np.array([env.observation_space.low[2],-1.0])
    high_state = np.array([env.observation_space.high[2],1.0])

    agent = SARSA(num_state=args.num_state, num_action=args.num_action, low_state=low_state, high_state=high_state)

    train(args, env, agent)
    test(args, env, agent)


if __name__ == "__main__":
    main()


