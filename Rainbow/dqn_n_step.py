from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple,Deque
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import gymnasium as gym

import sys

from numpy.ma.core import indices


class ReplayBuffer:
    def __init__(self, obs_dim:int, size:int, batch_size:int=32,n_step:int=3,gamma:float=0.99):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.max_size = size
        self.batch_size = batch_size
        self.ptr=0
        self.size=0

        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma=gamma
        self.n_step=n_step

    def store(self, obs:np.ndarray, act:np.ndarray, rew:float, next_obs:np.ndarray, done:bool)->Tuple[np.ndarray, np.ndarray, float, np.ndarray,bool]:
        transition = (obs, act,rew, next_obs, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return ()

        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer,self.gamma)
        obs,act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],next_obs=self.next_obs_buf[idxs],acts= self.acts_buf[idxs],rews=self.rews_buf[idxs],done=self.done_buf[idxs],indices=idxs)

    def sample_batch_from_idxs(self,indices:np.ndarray)->Dict[str, np.ndarray]:
        return dict(obs = self.obs_buf[indices],next_obs = self.next_obs_buf[indices],acts= self.acts_buf[indices],rews = self.rews_buf[indices],done = self.done_buf[indices])

    def __len__(self) -> int:
        return self.size

    def _get_n_step_info(self,n_step_buffer:Deque, gamma:float)->Tuple[np.int64, np.ndarray, bool]:
        rew,next_obs,done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rew = r + gamma*rew*(1-d)
            next_obs, done = (n_o,d) if d else (next_obs,done)

        return rew, next_obs, done





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
    def __init__(self, env:gym.Env, memory_size:int, batch_size:int, target_update:int, epsilon_decay:float, seed:int, max_epsilon:float=1.0, min_epsilon:float=0.1,gamma:float=0.99,n_step:int=3):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=1, gamma=gamma)
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

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

        if not self.is_test:
            self.transition+=[reward,next_state,done]

            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            else:
                one_step_transition = self.transition
            if one_step_transition:
                self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self)->torch.Tensor:
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self._compute_dqn_loss(samples,self.gamma)

        if self.use_n_step:
            samples = self.memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma**self.n_step
            n_loss = self._compute_dqn_loss(samples,gamma)
            loss = loss + n_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self,num_frames:int, plotting_interval:int=200):
        self.is_test=False
        state, _ = self.env.reset(seed=self.seed)
        update_cnt=0
        epsilons=[]
        losses=[]
        scores=[]
        score=0

        for frame_idx in range(1, num_frames+1):
            action = self.select_action(state)
            next_state, reward, done= self.step(action)

            state = next_state
            score += reward

            if done:
                state,_ = self.env.reset(seed=self.seed)
                scores.append(score)
                score=0

            if len(self.memory) >= self.batch_size:
                loss=self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon =max(self.min_epsilon,self.epsilon-(self.max_epsilon-self.min_epsilon)*self.epsilon_decay)
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx,scores,losses,epsilons)
        self.env.close()

    def test(self, video_folder:str)->None:
        self.is_test=True

        naive_env=self.env
        self.env = gym.wrappers.RecordVideo(self.env,video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done=False
        score=0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

        print("score:",score)
        self.env.close()

        self.env = naive_env


    def _compute_dqn_loss(self,samples: Dict[str, np.ndarray],gamma:float) -> torch.Tensor:
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)

        next_q_value = self.dqn_target(next_state).max(dim=1,keepdim=True)[0].detach()
        mask=1-done
        target = (reward + gamma * next_q_value*mask).to(device)
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()

env = gym.make("CartPole-v1", max_episode_steps=200, render_mode='rgb_array')

'''
print(env.observation_space.shape)
print(env.action_space.n)
print( env.action_space.sample())
print( env.action_space.sample())

sys.exit()
'''

seed=777
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
np.random.seed(seed)
seed_torch(seed)


num_frames=10000
memory_size=1000
batch_size=32
target_update=100
epsilon_decay=1/2000

agent = DQNAgent(env,memory_size,batch_size,target_update,epsilon_decay,seed)

agent.train(num_frames)

video_folder="videos/dqn_n_step"
agent.test(video_folder)






