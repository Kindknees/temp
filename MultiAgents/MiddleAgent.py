import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuralNetworks.LinearModels import Actor, CriticNetwork
from collections import deque, namedtuple
import os


class FixedSubPicker(object):
    def __init__(self, masked_sorted_sub, **kwargs):
        self.masked_sorted_sub = masked_sorted_sub
        self.subs_2_act = []
        n_subs = len(masked_sorted_sub)
        self.count = np.zeros((n_subs, n_subs), int)
        self.previous_sub = -1

    def complete_reset(self):
        self.subs_2_act = []
        self.previous_sub = -1

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # randomize the order in which the substations are activated
            self.subs_2_act = list(self.masked_sorted_sub)
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act, (None, None)

    def count_transitions(self, next_sub):
        if self.previous_sub >= 0:
            prev = np.flatnonzero(self.masked_sorted_sub == self.previous_sub).squeeze()
            next = np.flatnonzero(self.masked_sorted_sub == next_sub).squeeze()
            self.count[prev, next] += 1

    @property
    def transition_probs(self):
        row_sums = self.count.sum(axis=1, keepdims=True)
        non_zero_rows = (row_sums != 0).squeeze()
        probs = np.zeros_like(self.count, float)
        probs[non_zero_rows] = self.count[non_zero_rows] / row_sums[non_zero_rows]
        return probs


class RandomOrderedSubPicker(FixedSubPicker):
    def __init__(self, masked_sorted_sub, **kwargs):
        super().__init__(masked_sorted_sub)

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # randomize the order in which the substations are activated
            self.subs_2_act = list(self.masked_sorted_sub)
            random.shuffle(self.subs_2_act)
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act, (None, None)


class RuleBasedSubPicker(FixedSubPicker):
    def __init__(self, masked_sorted_sub, action_space):
        super().__init__(masked_sorted_sub)
        self.sub_line_or = []
        self.sub_line_ex = []
        for sub in self.masked_sorted_sub:
            self.sub_line_or.append(
                np.flatnonzero(action_space.line_or_to_subid == sub)
            )
            self.sub_line_ex.append(
                np.flatnonzero(action_space.line_ex_to_subid == sub)
            )

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
            # which applies an action to substations that require urgent care.
            rhos = []
            for sub in self.masked_sorted_sub:
                sub_i = np.flatnonzero(self.masked_sorted_sub == sub).squeeze()
                rho = np.append(
                    obs.rho[self.sub_line_or[sub_i]].copy(),
                    obs.rho[self.sub_line_ex[sub_i]].copy(),
                )
                rho[rho == 0] = 3
                rho_max = rho.max()
                rho_mean = rho.mean()
                rhos.append((rho_max, rho_mean))
            order = sorted(
                zip(self.masked_sorted_sub, rhos), key=lambda x: (-x[1][0], -x[1][1])
            )
            self.subs_2_act = list(list(zip(*order))[0])
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act

class PPOMiddleAgent:
    def __init__(self, subs, action_space, state_dim=128, hidden_dim=64, device='cpu', **kwargs):
        """
        A PPO-based middle agent that selects substations using a neural network.
        
        Args:
            subs (list): List of substation IDs managed by the agent.
            action_space: The action space from the environment.
            state_dim (int): Dimension of the input state.
            hidden_dim (int): Dimension of hidden layers in the neural network.
            device (str): Device to use ('cpu' or 'cuda').
            **kwargs: Additional arguments (e.g., middle_lr, middle_batch_size, middle_epsilon).
        """
        self.subs = subs
        self.n_sub = len(subs)  # Compute number of substations
        self.action_space = action_space
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = kwargs.get('middle_lr', 3e-4)
        self.batch_size = kwargs.get('middle_batch_size', 64)
        self.epsilon = kwargs.get('middle_epsilon', 0.2)
        self.gamma = kwargs.get('middle_gamma', 0.995)
        self.lmbda = kwargs.get('middle_lambda', 0.95)
        
        # Neural networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_sub),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)
        
        # Memory for PPO updates
        self.memory = deque(maxlen=10000)
        self.update_step = 0

    def complete_reset(self):
        """Reset the agent's memory and state."""
        self.memory.clear()

    def _get_state(self, state):
        """Convert observation or state to a tensor of shape (state_dim,)."""
        # Assume state is a tensor from IMARL.get_current_state()
        state = state.squeeze()
        if state.shape[0] < self.state_dim:
            state = torch.cat([state, torch.zeros(self.state_dim - state.shape[0], device=self.device)])
        elif state.shape[0] > self.state_dim:
            state = state[:self.state_dim]
        return state

    def pick_sub(self, state, sample):
        """Select a substation using the PPO policy.
        
        Args:
            state (torch.Tensor): The current state (from IMARL.get_current_state()).
            sample (bool): Whether to sample from the policy (True) or take the argmax (False).
        
        Returns:
            tuple: (sub_id, (action_idx, log_prob)) where sub_id is the selected substation ID,
                   action_idx is the index in self.subs, and log_prob is the log probability.
        """
        state = self._get_state(state).to(self.device)
        probs = self.actor(state.unsqueeze(0))
        if sample:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        else:
            action_idx = torch.argmax(probs, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action_idx)
        sub_id = self.subs[action_idx.item()]
        return sub_id, (action_idx.item(), log_prob)

    def critic(self, state, adj=None):
        """Compute the value function for the given state.
        
        Args:
            state (torch.Tensor): The current state.
            adj (torch.Tensor, optional): Adjacency matrix (ignored).
        
        Returns:
            torch.Tensor: The value estimate.
        """
        state = self._get_state(state).to(self.device)
        return self.critic_net(state.unsqueeze(0))

    def save_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Save a transition for PPO updates.
        
        Args:
            state (torch.Tensor): The current state.
            action (int): The action index (substation index).
            reward (float): The reward received.
            next_state (torch.Tensor): The next state.
            done (bool): Whether the episode is done.
            log_prob (torch.Tensor): Log probability of the action.
            value (float): Value estimate for the state.
        """
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def update(self):
        """Perform a PPO update if enough transitions are available."""
        if len(self.memory) < self.batch_size:
            return
        self.update_step += 1
        # Collect batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([t[0] for t in batch]).to(self.device)
        actions = torch.tensor([t[1] for t in batch], device=self.device)
        rewards = torch.tensor([t[2] for t in batch], device=self.device)
        next_states = torch.stack([t[3] for t in batch]).to(self.device)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=self.device)
        old_log_probs = torch.stack([t[5] for t in batch]).to(self.device)
        old_values = torch.tensor([t[6] for t in batch], device=self.device)
        
        # Compute advantages
        next_values = self.critic_net(next_states).squeeze()
        advantages = []
        returns = []
        gae = 0
        for r, d, v, nv in zip(reversed(rewards), reversed(dones), reversed(old_values), reversed(next_values)):
            delta = r + self.gamma * nv * (1 - d) - v
            gae = delta + self.gamma * self.lmbda * (1 - d) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic_net(states).squeeze(), returns)
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def save_model(self, path, name):
        """Save the actor and critic models."""
        torch.save(self.actor.state_dict(), os.path.join(path, f'{name}_actor.pt'))
        torch.save(self.critic_net.state_dict(), os.path.join(path, f'{name}_critic.pt'))

    def load_model(self, path, name):
        """Load the actor and critic models."""
        self.actor.load_state_dict(torch.load(os.path.join(path, f'{name}_actor.pt')))
        self.critic_net.load_state_dict(torch.load(os.path.join(path, f'{name}_critic.pt')))

    def load_mean_std(self, mean, std):
        """Load mean and std for state normalization."""
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)