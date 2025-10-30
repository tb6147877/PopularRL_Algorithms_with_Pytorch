import os


from pettingzoo.butterfly import knights_archers_zombies_v10
import supersuit as ss
import numpy as np
import torch as th
from torch import Tensor
from torch import nn
from torch.distributions import Categorical
import abc
from torch.nn import functional as F
from pettingzoo import AECEnv

class Critic(nn.Module, abc.ABC):
    '''PPO Base Critic Module'''
    def __init__(self, statesize:int, batch_size:int=64, epochs:int=10, lr:float=0.0001):
        '''
                Args:
                    statesize (int): State space size (size of first layer input).
                    batch_size (int, optional): Batch size for training. Defaults to 64.
                    epochs (int): How many times we update when training.
                    lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.0003.
                '''
        super().__init__()
        self.ssize, self.epochs, self.batch_size = statesize, epochs, batch_size

        self.value_net_in = nn.Sequential(
            nn.Linear(statesize, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        self.value_net_out = nn.Linear(512, 1)

        self.optim = th.optim.Adam(self.parameters(), lr)

    def compute_gae(self, rewards:Tensor, values: Tensor, last_value:Tensor, gamma:float, lam:float)->tuple[Tensor, Tensor]:
        """Compute generalized advantage estimation (GAE) for a given episode.

                Args:
                    rewards: tensor containing rewards
                    values: tensor containing value estimates
                    last_value: value estimate for last timestep
                    gamma: discount factor
                    lam: lambda

                Returns:
                    advantages: tensor containing advantages
                    returns: tensor containing returns = advantages + values
                """
        horizon = len(rewards)
        advantages, returns = th.zeros_like(rewards), th.zeros_like(rewards)
        next_value = last_value
        next_advantage = 0.0

        # compute advantages and returns in backward
        for t in reversed(range(horizon)):
            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]

            # GAE = A_t = delta_t + (gamma * lambda) * A_{t+1}
            advantages[t] = delta + gamma*lam*next_advantage

            next_advantage = advantages[t]
            next_value = values[t]

            returns[t] = advantages[t] + values[t]

        return advantages, returns

class DecentralizedCritic(Critic):
    '''Decentralized PPO Critic Module. Used in `IPPO`.'''
    def forward(self, state:th.Tensor)->th.Tensor:
        out = self.value_net_in(state)
        return self.value_net_out(out)

    def value(self, state:th.Tensor):
        state = state.flatten(start_dim=-2)
        return self.forward(state)

    def update(self, dataset_agent:dict):
        avgloss=0.0
        n_batches = 0
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            states = dataset_agent["states"]
            returns = dataset_agent["returns"]

            dataset_size = states.shape[0]

            # Shuffle the entire dataset at the start of each epoch
            permutation = th.randperm(dataset_size)

            # Process the entire dataset in mini-batches
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = permutation[start_idx:end_idx]

                # Get mini-batch data
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]

                # Make predictions and compute loss
                predictions = self.value(batch_states).squeeze(1)
                loss = F.mse_loss(predictions, batch_returns)

                # Update model
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Track loss
                epoch_loss += loss.item()
                epoch_batches += 1

            avgloss += epoch_loss
            n_batches += epoch_batches

        # Print average loss (total loss / total number of batches)
        return avgloss/n_batches

    def add_gae(self, agent_trajectories:list, gamma:float=0.99, lam:float=0.95)->dict:
        # Process each agent's trajectories
        for trajectory in agent_trajectories:
            states = trajectory["states"]
            rewards = trajectory["rewards"]

            with th.no_grad():
                values = self.value(states).squeeze(-1)

                # For the last state, we need to estimate its value for bootstrapping
                # If this is a terminal state, last_value should be 0
                # However, in practice, we're going to use the critic's estimate
                last_value = th.zeros(1)
                if len(states)>0:
                    last_value = self.value(states[-1:]).squeeze(-1)

            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, last_value, gamma, lam)

            # Add to trajectory
            trajectory["advantages"] = advantages
            trajectory["rtgs"] = returns
        return agent_trajectories


class Actor(nn.Module):
    def __init__(self, statesize:int, n_actions:int, epochs:int, batch_size:int=64, eps:float=0.2, c_ent:float=0.0, lr:float=0.0003):
        '''
                Args:
                    statesize (int): State space size (size of first layer input).
                    n_actions (int): Number of discrete actions the agent should output.
                    epochs (int): How many times we update when training.
                    batch_size (int, optional): Batch size for training. Defaults to 64.
                    eps (float, optional): Epsilon in PPO^CLIP objective. Defaults to 0.2.
                    c_ent (float, optional): Entropy coefficient. Defaults to 0.
                    lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.0003.
                '''
        super().__init__()
        self.ssize, self.epochs, self.batch_size, self.eps, self.c_ent = statesize, epochs, batch_size, eps, c_ent
        self.policy_net = nn.Sequential(
            nn.Linear(statesize, 512),nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )

        self.action_net = nn.Linear(512,n_actions)

        self.optim = th.optim.Adam(self.parameters(), lr)

    def forward(self, state:np.array)->Tensor:
        '''Compute logits for a given game state.'''
        if not isinstance(state, Tensor): state = th.as_tensor(state, dtype=th.float32)
        if not state.shape[-1] == self.ssize: state = th.flatten(state, start_dim=-2)
        logits = self.action_net(self.policy_net(state))
        return logits

    def distribution(self, state:np.array)->Categorical:
        '''Outputs a probability distribution over discrete actions for a given game state'''
        logits = self.forward(state)
        action_dist = Categorical(logits=logits)
        return action_dist

    def action(self, state:np.array)->tuple[int, th.Tensor]:
        '''Output a pair of (action, logprob) for a given game state.'''
        actions_dist = self.distribution(state)
        action = actions_dist.sample()
        logprob = actions_dist.log_prob(action)
        return action.item(), logprob

    def objective(self, dataset:dict, indices)->th.Tensor:
        '''PPO Objective Function'''
        states = dataset['states'][indices]
        actions = dataset['actions'][indices]
        logprobs_old = dataset['logprobs'][indices]
        advantages = dataset['advantages'][indices]

        distributions = self.distribution(states)

        entropy = distributions.entropy().mean()
        loss_entropy = self.c_ent * entropy

        logprobs_new = distributions.log_prob(actions)
        ratio = th.exp(logprobs_new - logprobs_old)
        clipped = th.clip(ratio, 1.0-self.eps, 1.0+self.eps)*advantages
        loss_clip = th.min(ratio*advantages, clipped).mean()

        return loss_clip + loss_entropy

    def update(self, dataset:dict):
        '''Train the actor.'''
        dataset_size = dataset["states"].shape[0]

        for epoch in range(self.epochs):
            # shuffle the start of each epoch
            permutation = th.randperm(dataset_size)

            # process entire dataset
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = permutation[start_idx:end_idx]

                #update
                loss = -self.objective(dataset,batch_indices)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

class Algorithm(nn.Module, abc.ABC):
    '''Base Class for IPPO & MAPPO.'''
    def __init__(self, env:AECEnv, batch_size:int=64, epochs:int=10, eps:float=0.2, c_ent:float=0):
        '''
                Args:
                    env (AECEnv): _description_
                    batch_size (int, optional): _description_. Defaults to 64.
                    epochs (int, optional): _description_. Defaults to 10.
                    eps (float, optional): _description_. Defaults to 0.2.
                    c_ent (float, optional): _description_. Defaults to 0.
                '''
        super().__init__()

        env.reset()
        self.env = env
        self.ssize = np.prod(env.observation_space(env.agents[0]).shape)

        self.actors = nn.ModuleDict(
            {
                agent:Actor(statesize=self.ssize, n_actions=env.action_space(agent).n,batch_size=batch_size,eps=eps,epochs=epochs,c_ent=c_ent) for agent in env.agents
            }
        )

    def learn(self, niters:int, nsteps:int=2048, checkpoints_path:str="checkpoints", eval_path:str="evaluations"):
        '''Train the algorithm for `niters` iterations.

                Args:
                    niters (int): How many updates to perform.
                    nsteps (int, optional): How many steps to collect before performing an update. Defaults to 2048.
                    checkpoints_path (str, optional): Path to save algorithm state. Defaults to 'checkpoints'.
                '''

        # create directory for checkpoints if not exists
        try:os.mkdir(checkpoints_path)
        except:pass

        evaluations = {'mean_return':[], 'std_return':[], 'mean_eplen':[]}

        for i in range(niters):
            # collect trajectories
            trajectories = self.collect_trajectories(nsteps)
            trajectories = self.split_trajectories(trajectories)

            # compute GAE and returns
            trajectories = self.add_gae(trajectories, gamma=0.99, lam=0.95)

            # flatten trajectories before training
            flattened = self.flatten_trajectories(trajectories)

            # updates critic and actors
            loss_critics = self.update_critic(flattened)
            self.update_actors(flattened)

            # eval
            with th.no_grad():
                mean_return, std_return, mean_length = self.evaluate(10)
                evaluations['mean_return'].append(mean_return)
                evaluations['mean_eplen'].append(mean_length)
                evaluations['std_return'].append(std_return)

                log = f'¤ {i + 1} / {niters}: Reward={mean_return:.1f}, ' + \
                      f'Std={std_return:.1f}, Length={mean_length:.0f}, LossCritic={loss_critics:.3f}'

                print(log, flush=True)

            # save checkpoint
            th.save(self.state_dict(), f'{checkpoints_path}/ippo{i + 1}_{mean_return:.1f}_{std_return:.1f}.pth')

            # save logs
            #np.save(eval_path, np.array(evaluations))

    def update_actors(self, flattened:dict):
        '''Update the actors one by one by calling `actor.update` using their own collected data.'''
        for agent, actor in self.actors.items():
            dataset_agent = flattened[agent]
            actor.update(dataset_agent)

    def split_trajectories(self, trajectories:dict)->dict:
        return {agent:[self.split_trajectory(trajectory) for trajectory in trajs] for agent, trajs in trajectories.items()}

    def evaluate(self, N:int=10):
        '''Evaluate the algorithm for `N` episodes.'''
        with th.no_grad():
            self.eval()
            n_agents = len(self.actors.keys())
            seeds = list(range(N))
            returns, lengths = [], []

            for i in range(N):
                self.env.reset(seed=seeds[i])
                return_=length = 0.0

                for agent in self.env.agent_iter():
                    obs, reward, term, trunc, _ = self.env.last()
                    return_+=reward

                    if term or trunc:
                        # if agent is dead or max_cycles is reached
                        action = None
                        self.env.step(action)
                        continue

                    # get the corresponding actor
                    actor = self.actors[agent]
                    # get the action
                    action, _ = actor.action(obs)
                    # take a step to get next state
                    self.env.step(action)
                    length += 1/n_agents

                returns.append(return_);lengths.append(length)

            self.train()
            return np.mean(returns), np.std(returns), np.mean(lengths)


    @abc.abstractmethod
    def collect_trajectories(self, N:int)->dict:
        '''Collect `N` trajectories for each agent in the environment.
                Trajectory = (state1, action1, reward1, logprob1, state2, ...).
                '''
        pass

    @abc.abstractmethod
    def split_trajectory(self, trajectory:list)->dict[str, Tensor]:
        '''Split a trajectory into separate tensors of same elements.'''
        pass

    @abc.abstractmethod
    def flatten_trajectories(self, trajectories:dict):
        '''Flatten the collected trajectories.

                `collect_trajectories` returns a dict with, for each agent, N collected trajectories.
                This function flatten the N trajectories into one "big" trajectory, such that we can then sample minbatches.
                '''
        pass

    @abc.abstractmethod
    def add_gae(self, trajectories, gamma=0.99, lam=0.95):
        '''Add Generalized Advantage Estimations to collected trajectories.'''
        pass

    @abc.abstractmethod
    def update_critic(self, flattened:dict):
        '''Update the critic(s).'''
        pass

class IPPO(Algorithm):
    '''Independant Proximal Policy Optimization

       Decentralized Training - Decentralized Execution Framework:
       N Critics for N Actors
       '''
    def __init__(self, env:AECEnv, batch_size:int=64, epochs:int=10, eps:float=0.2, c_ent:float=0):
        super().__init__(env, batch_size, epochs, eps, c_ent)

        self.critics = nn.ModuleDict({
            agent:DecentralizedCritic(statesize=self.ssize, batch_size=batch_size, epochs= epochs) for agent in env.agents
        })

    def flatten_trajectories(self, trajectories:dict):
        flattened = {}
        for agent, trajectories in trajectories.items():
            flattened[agent] = {
                'states':th.cat([t['states'] for t in trajectories]),
                'actions': th.cat([t['actions'] for t in trajectories]),
                'rewards': th.cat([t['rewards'] for t in trajectories]),
                'logprobs': th.cat([t['logprobs'] for t in trajectories]),
                'advantages': th.cat([t['advantages'] for t in trajectories]),
                'returns': th.cat([t['rtgs'] for t in trajectories]),
            }
        return flattened

    def add_gae(self, trajectories:dict, gamma, lam):
        for agent in trajectories.keys():
            critic:DecentralizedCritic
            critic = self.critics[agent]
            trajectories[agent] = critic.add_gae(trajectories[agent])

        return trajectories

    def update_critic(self, flattened:dict):
        L=0.0
        for agent, critic in self.critics.items():
            dataset_agent = flattened[agent]
            L+=critic.update(dataset_agent)
        return L/len(flattened.keys())

    def collect_trajectories(self, N:int)->dict:
        trajectories = {agent:[] for agent in self.actors.keys()}
        total_length = 0
        n_agents = len(self.actors.keys())

        while total_length < N:

            for t in trajectories.values():t.append([])

            self.env.reset()

            for agent in self.env.agent_iter():
                obs, reward, term, trunc, _ = self.env.last()

                if term or trunc:
                    # if agent is dead or max_cycles is reached
                    action = None
                    self.env.step(action)
                    continue

                # get the corresponding actor
                actor = self.actors[agent]
                # get the action, logprob pair
                action, logprob = actor.action(obs)
                # add the items to the current trajectory
                elements = [obs, action, reward, logprob]
                trajectories[agent][-1].extend(elements)
                # take a step to get next state
                self.env.step(action)

                total_length+=1/n_agents
        return trajectories

    def split_trajectory(self, trajectory:list)->dict[str, th.Tensor]:
        states =[]; rewards=[]; actions=[]; logprobs=[]
        for i in range(0, len(trajectory), 4): # elements = [obs, action, reward, logprob]
            s,a,r,l = trajectory[i:i+4]
            states.append(s); rewards.append(r); actions.append(a); logprobs.append(l)
        return {
            'states':th.as_tensor(np.array(states), dtype=th.float32),
            'rewards': th.as_tensor(rewards, dtype=th.float32),
            'actions': th.as_tensor(actions, dtype=th.int64),
            'logprobs': th.as_tensor(logprobs, dtype=th.float32),
        }


def create_environment(num_agents=2, max_cycles=2500, render_mode = None):
    env = knights_archers_zombies_v10.env(
        max_cycles=max_cycles, num_archers=num_agents,
        num_knights=0, max_zombies=4,
        vector_state=True, render_mode=render_mode)

    return ss.black_death_v3(env)

if __name__ == '__main__':
    env = create_environment()
    algo = IPPO(env, batch_size=256, c_ent=0.01)

    print(algo)

    print('\nStarting to train IPPO algorithm...')

    algo.learn(niters=10, nsteps=8192)