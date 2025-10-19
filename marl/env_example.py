from pettingzoo.mpe import simple_spread_v3
import time

import imageio.v2 as imageio

env = simple_spread_v3.env(N=2,local_ratio = 0.5,max_cycles=25,continuous_actions=False, render_mode = "rgb_array")

num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).n
observation_size = env.observation_space(env.possible_agents[0]).shape


print(f"{num_agents} agents")
frames = []
for i in range(num_agents):
    num_actions = env.action_space(env.possible_agents[i]).n
    observation_size = env.observation_space(env.possible_agents[i]).shape
    print(i, env.possible_agents[i], "num_actions:", num_actions, "observation_size:", observation_size)

total_reward=0
for epoch in range(1):
    env.reset()
    for i, agent in enumerate(env.agent_iter()):
        observation, reward, terminated, truncated, info = env.last()

        done = terminated or truncated


        frame = env.render()
        frames.append(frame)

        if env.agent_selection == "agent_0":
            total_reward += reward

        action = 0
        if done:
            break

        action = env.action_space(agent).sample()
        env.step(action)

        print(i, agent)
        print(f"action={action}, observation={observation}, reward={reward}, done={done}, info={info}")

        #time.sleep(3)
    print(f"total_reward={total_reward}")

imageio.mimsave("simple_spread.mp4", frames, fps=20)
env.close()