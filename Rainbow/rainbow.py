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

from segment_tree import MinSegmentTree, SumSegmentTree
import sys
import math


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

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim:int, size:int, batch_size:int=32, alpha:float=0.6,n_step:int=1,gamma:float=0.99):
        assert alpha >= 0
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size,n_step,gamma)
        self.max_priority = 1.0
        self.tree_ptr=0
        self.alpha = alpha

        tree_capacity=1
        while tree_capacity<self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs:np.ndarray, act:int, rew:float, next_obs:np.ndarray, done: bool)->Tuple[np.ndarray, np.ndarray, float, np.ndarray,bool]:
        transation = super().store(obs, act, rew, next_obs, done)
        if transation:
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha

            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transation

    def sample_batch(self,beta:float=0.4) -> Dict[str, np.ndarray]:
        assert len(self)>=self.batch_size
        assert beta >= 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i,beta) for i in indices])

        return dict(obs=obs,next_obs=next_obs,acts=acts,rews=rews,done=done,weights=weights,indices=indices)

    def update_priorities(self, indices:List[int], priorities:np.ndarray):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)


    def _sample_proportional(self)->List[int]:
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment*i
            b = segment*(i+1)

            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx:int,beta:float):
        p_min=self.min_tree.min()/self.sum_tree.sum()
        max_weight=(p_min*len(self))**(-beta)

        p_sample = self.sum_tree[idx]/self.sum_tree.sum()
        weight = (p_sample*len(self))**(-beta)
        weight = weight/max_weight

        return weight

class NoisyLinear(nn.Module):
    def __init__(self,in_features:int, out_features:int,std_init:float=0.5):
        super(NoisyLinear,self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init=std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon",torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range=1/math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range,mu_range)
        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x,self.weight_mu+self.weight_sigma*self.weight_epsilon,self.bias_mu+self.bias_sigma*self.bias_epsilon)

    @staticmethod
    def scale_noise(size:int)->torch.Tensor:
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Network(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, atom_size:int, support: torch.Tensor):
        super(Network, self).__init__()

        self.support=support
        self.atom_size=atom_size
        self.out_dim=out_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim*atom_size)

        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        dist=self.dist(x)
        q = torch.sum(dist*self.support, dim=2)
        return q

    def dist(self, x:torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1,self.out_dim,self.atom_size)
        value = self.value_layer(val_hid).view(-1,1,self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist


    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()



class DQNAgent:
    def __init__(self, env:gym.Env, memory_size:int, batch_size:int, target_update:int,  seed:int, gamma:float=0.99,alpha:float=0.2,beta:float=0.6,prior_eps:float=1e-6, v_min:float=0.0, v_max:float=200.0,atom_size:int=51,n_step:int=3):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.seed = seed
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.beta=beta
        self.prior_eps=prior_eps
        self.memory = PrioritizedReplayBuffer(obs_dim,memory_size,batch_size,alpha=alpha,gamma=gamma)

        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)


        self.dqn = Network(obs_dim, action_dim,self.atom_size,self.support).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim,self.atom_size,self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()

        self.is_test=False

    def select_action(self, state:np.ndarray) -> np.ndarray:

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
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self)->torch.Tensor:
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1,1)).to(self.device)
        indices = samples["indices"]



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





    def _compute_dqn_loss(self,samples: Dict[str, np.ndarray],gamma:float) -> torch.Tensor:
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + self.gamma * self.support * (1 - done)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.floor().long() + 1

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long().unsqueeze(1).expand(
                    self.batch_size, self.atom_size).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u.clamp(max=self.atom_size - 1) + offset).view(-1),
                                          (next_dist * (b - l.float())).view(-1))

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])

        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
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

video_folder="videos/rainbow"
agent.test(video_folder)






