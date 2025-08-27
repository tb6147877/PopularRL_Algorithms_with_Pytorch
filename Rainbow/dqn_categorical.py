import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import gymnasium as gym

import sys


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
    def __init__(self, in_dim:int, out_dim:int, atom_size:int, support:torch.Tensor):
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim*atom_size)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q=torch.sum(dist*self.support,dim=2)
        return q

    def dist(self,x:torch.Tensor)->torch.Tensor:
        q_atoms = self.layers(x).view(-1,self.out_dim,self.atom_size)
        dist = F.softmax(q_atoms,dim=-1)
        dist = dist.clamp(min=1e-3) #for avoiding nans

        return dist




class DQNAgent:
    def __init__(self, env:gym.Env, memory_size:int, batch_size:int, target_update:int, epsilon_decay:float, seed:int, max_epsilon:float=1.0, min_epsilon:float=0.1,gamma:float=0.99, v_min:float=0.0, v_max:float=200.0,atom_size:int=51):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

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

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.dqn = Network(obs_dim, action_dim,atom_size,self.support).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim,atom_size,self.support).to(self.device)
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
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self)->torch.Tensor:
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)
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





    def _compute_dqn_loss(self,samples: Dict[str, np.ndarray]) -> torch.Tensor:
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn_target(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + self.gamma * self.support * (1 - done)
            t_z= t_z.clamp(min=self.v_min,max = self.v_max)
            b = (t_z-self.v_min)/delta_z
            l = b.floor().long()
            u = b.floor().long()+1

            offset = (torch.linspace(0,(self.batch_size-1)*self.atom_size,self.batch_size).long().unsqueeze(1).expand(self.batch_size,self.atom_size).to(self.device))

            proj_dist = torch.zeros(next_dist.size(),device=self.device)
            proj_dist.view(-1).index_add_(0,(l+offset).view(-1), (next_dist*(u.float()-b)).view(-1))
            proj_dist.view(-1).index_add_(0,(u.clamp(max=self.atom_size-1)+offset).view(-1),(next_dist*(b-l.float())).view(-1))

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size),action])

        loss = -(proj_dist*log_p).sum(1).mean()

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

video_folder="videos/dqn_categorical"
agent.test(video_folder)






