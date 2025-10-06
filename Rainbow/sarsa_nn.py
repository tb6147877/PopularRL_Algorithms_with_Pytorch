import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import  torch

import random
import numpy as np
import matplotlib.pyplot as plt
import os



class QNet(nn.Module):
    def __init__(self, dim_state, num_actions):
        super().__init__()

        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)

        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

class SARSA:
    def __init__(self, dim_state, num_actions, gamma=0.99):
        self.gamma = gamma
        self.Q = QNet(dim_state, num_actions)
        self.target_Q = QNet(dim_state, num_actions)

        self.target_Q.load_state_dict(self.Q.state_dict())

    def get_action(self, state):
        qvals = self.Q(state)
        return qvals.argmax(dim=-1)

    def compute_loss(self, args, s_list, a_list, r_list):
        batch_s = np.array(s_list)
        batch_s = torch.tensor(batch_s,dtype=torch.float32)

        batch_a = np.array(a_list)
        batch_a = torch.tensor(batch_a, dtype=torch.long)

        num = len(r_list)
        state = batch_s[:num-args.m+1,:]
        action = batch_a[:num-args.m+1]
        qvals = self.Q(state).gather(1, action.unsqueeze(1)).squeeze()

        R=0
        for i in reversed(range(num-args.m,num)):
            R = R * args.gamma+r_list[i]

        rewards = [R]
        for i in reversed(range(num-args.m)):
            R-=args.gamma**(args.m-1)*r_list[i+args.m]
            R = R*args.gamma + r_list[i]
            rewards.append(R)
        rewards.reverse()
        rewards = torch.tensor(rewards,dtype=torch.float32)

        with torch.no_grad():
            state = batch_s[args.m:,:]
            action = batch_a[args.m:]

            m_step_qvals =self.target_Q(state).gather(1, action.unsqueeze(1)).squeeze()

            target_values = args.gamma**args.m*m_step_qvals
            target_values = torch.cat([target_values,torch.tensor([0.0])])
            target_values = target_values +rewards

        td_delta = qvals - target_values

        value_loss = td_delta.square().mean()

        return value_loss

    def soft_update(self, tau):
        for target_param, param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)

def train(args,env,agent):
    optimizer = torch.optim.Adam(agent.Q.parameters(), lr=args.lr)







