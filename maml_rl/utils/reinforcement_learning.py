import torch
import numpy as np
from maml_rl.utils.torch_utils import weighted_mean, to_numpy

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    delta = np.inf
    while delta >= theta:
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        new_values = np.max(q_values, axis=1)
        delta = np.max(np.abs(new_values - values))
        values = new_values

    return values

def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values

def get_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

import torch
import numpy as np

def reinforce_loss(policy, episodes, params=None):
    # Compute log probabilities of the actions taken
    pi = policy(episodes.observations, params=params)
    log_probs = pi.log_prob(episodes.actions)
    
    if log_probs.dim() > 2:
        log_probs = log_probs.sum(dim=2)
        
    loss = -weighted_mean(log_probs * episodes.advantages, 
                          lengths=episodes.lengths)
    return loss.mean()

def weighted_mean(tensor, lengths=None):
    if lengths is None:
        return tensor.mean()
    if tensor.dim() < 2:
        return tensor.mean()
        
    batch_size = tensor.shape[0]
    max_len = tensor.shape[1]
    mask = torch.arange(max_len, device=tensor.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    masked_tensor = tensor * mask.float()
    return masked_tensor.sum() / mask.sum()

def get_returns(episodes):
    return episodes.rewards.sum(dim=1)

class Episode:
    def __init__(self, observation, action, reward, done):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        
        self._advantages = None
        self._returns = None

    def append(self, observation, action, reward, next_observation, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dones.append(done)

    def compute_advantages(self, baseline, gamma=0.95, gae_lambda=1.0):
        observations = torch.from_numpy(np.array(self.observations)).float()
        rewards = torch.from_numpy(np.array(self.rewards)).float()
        
        with torch.no_grad():
            values = baseline(observations).squeeze()
            
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(self.dones[-1])
                next_value = 0
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        self._advantages = advantages
        self._returns = advantages + values

    @property
    def length(self):
        return len(self.rewards)

class BatchEpisodes:
    def __init__(self, episodes, batch_size, device='cpu'):
        self.episodes = episodes
        self.batch_size = batch_size
        self.device = device
        
        self._observations = None
        self._actions = None
        self._rewards = None
        self._advantages = None
        self._returns = None
        self._mask = None

    @property
    def lengths(self):
        return torch.tensor([ep.length for ep in self.episodes], device=self.device)

    @property
    def observations(self):
        if self._observations is None:
            self._observations = self._collate([ep.observations for ep in self.episodes])
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            self._actions = self._collate([ep.actions for ep in self.episodes])
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            self._rewards = self._collate([ep.rewards for ep in self.episodes])
        return self._rewards

    @property
    def advantages(self):
        if self._advantages is None:
            self._advantages = self._collate([ep._advantages for ep in self.episodes])
        return self._advantages
        
    @property
    def returns(self):
        if self._returns is None:
            self._returns = self._collate([ep._returns for ep in self.episodes])
        return self._returns
        
    @property
    def mask(self):
        if self._mask is None:
            lengths = self.lengths
            max_len = lengths.max()
            self._mask = torch.arange(max_len, device=self.device).expand(self.batch_size, max_len) < lengths.unsqueeze(1)
            self._mask = self._mask.float()
        return self._mask

    def _collate(self, batch_list):
        max_len = max([len(item) for item in batch_list])
        
        first_item = batch_list[0]
        # Check if the first item is already a tensor (like in advantages) 
        # or a list/array (like in observations)
        if torch.is_tensor(first_item):
            # FIX: If input is a Tensor sequence (Time, ...), element shape is (...)
            # For 1D tensor (Time,), element shape is () (scalar)
            first_shape = first_item.shape[1:]
        else:
            first_item_elem = first_item[0]
            if isinstance(first_item_elem, np.ndarray):
                first_shape = torch.from_numpy(first_item_elem).shape
            elif not torch.is_tensor(first_item_elem):
                first_shape = torch.tensor(first_item_elem).shape
            else:
                first_shape = first_item_elem.shape
            
        shape = (len(batch_list), max_len) + first_shape
        padded_tensor = torch.zeros(shape, dtype=torch.float32, device=self.device)
        
        for i, item_seq in enumerate(batch_list):
            if torch.is_tensor(item_seq):
                seq_tensor = item_seq.float()
            elif isinstance(item_seq[0], np.ndarray):
                seq_tensor = torch.from_numpy(np.stack(item_seq)).float()
            elif isinstance(item_seq[0], torch.Tensor):
                seq_tensor = torch.stack(item_seq)
            else:
                seq_tensor = torch.tensor(item_seq).float()
                
            length = seq_tensor.shape[0]
            padded_tensor[i, :length] = seq_tensor.to(self.device)
            
        return padded_tensor

    def to(self, device):
        self.device = device
        if self._observations is not None:
            self._observations = self._observations.to(device)
        if self._actions is not None:
            self._actions = self._actions.to(device)
        if self._rewards is not None:
            self._rewards = self._rewards.to(device)
        if self._advantages is not None:
            self._advantages = self._advantages.to(device)
        if self._returns is not None:
            self._returns = self._returns.to(device)
        if self._mask is not None:
            self._mask = self._mask.to(device)
        return self