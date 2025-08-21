import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple

import gymnasium as gym

class ReplayBuffer:
    def __init__(self, obs_dim:int, size:int, batch_size:int=32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.max_size = size
        self.batch_size = batch_size
        self.ptr=0
        self.size=0

    def store(self, obs:np.ndarray, act:np.ndarray, rew:float, next_obs:np.ndarray, done:bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],next_obs=self.next_obs_buf[idxs],acts= self.acts_buf[idxs],rews=self.rews_buf[idxs],done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size




class Network(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNAgent:
    def __init__(self, env:gym.Env, memory_size:int, batch_size:int, target_update:int, epsilon_decay:float, seed:int, max_epsilon:float=1.0, min_epsilon:float=0.1,gamma:float=0.99):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval() # todo:这里要弄清楚

        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()

        self.is_test=False

    def select_action(self, state:np.ndarray) -> np.ndarray:
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition= [state, selected_action]

        return selected_action

    def step(self, action:np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state,reward,terminated,truncated,_ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test
            self.transition+=[reward,next_state,done]
            self.memory.store(*self.transition)

        return next_state, reward, done






