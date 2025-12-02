import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

# Assuming these imports and utilities exist in your framework
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.utils.networks import make_q_network, soft_update # Assume Q-net creation and soft update utility exists

class MAML_SAC(GradientBasedMetaLearner):
    """
    Model-Agnostic Meta-Learning (MAML) integrated with Soft Actor-Critic (SAC).

    The inner loop adaptation uses the SAC objective (Policy and Q-function updates).
    The outer loop performs the meta-gradient update based on the adapted policy's
    performance on validation data (maximizing return).
    """
    def __init__(self,
                 policy,
                 fast_lr=1e-3,
                 first_order=False,
                 outer_lr=3e-4,
                 sac_alpha=0.2, # Entropy regularization coefficient
                 gamma=0.99, # Discount factor
                 tau=0.005, # Soft update rate
                 q_hidden_sizes=[128, 128],
                 device='cpu'):
        
        super(MAML_SAC, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order
        self.sac_alpha = sac_alpha
        self.gamma = gamma
        self.tau = tau

        # --- Q-Network Setup (Managed by MAML-SAC for Inner Loop) ---
        # SAC requires two Q-networks and two target Q-networks.
        # We need a way to pass the task-specific adapted Q-network parameters.
        # For simplicity, we'll use a single set of networks here that are updated
        # in the outer loop, and we track the adapted parameters for the inner loop.
        
        input_size = sum(p.numel() for p in self.policy.parameters()) # Placeholder: actual input size needed

        # Assuming policy.obs_dim and policy.action_dim exist for Q-net construction
        obs_dim = self.policy.obs_dim
        action_dim = self.policy.action_dim

        self.qf1 = make_q_network(obs_dim, action_dim, q_hidden_sizes).to(self.device)
        self.qf2 = make_q_network(obs_dim, action_dim, q_hidden_sizes).to(self.device)
        self.target_qf1 = make_q_network(obs_dim, action_dim, q_hidden_sizes).to(self.device)
        self.target_qf2 = make_q_network(obs_dim, action_dim, q_hidden_sizes).to(self.device)

        # Initialize target networks to match Q-networks
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        # Outer loop optimizers (Meta-optimizers)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=outer_lr)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=outer_lr)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=outer_lr)

    async def adapt(self, train_futures, first_order=None):
        """
        Performs the inner loop SAC adaptation for a batch of tasks.
        
        NOTE: MAML + SAC is usually implemented with an off-policy sampler (PEARL)
        or by restricting MAML to on-policy SAC updates. This implementation
        assumes on-policy data is sampled, but the SAC objectives are used.
        """
        if first_order is None:
            first_order = self.first_order

        adapted_params_list = []
        qf_params_list = []

        # We must adapt Q-networks and Policy simultaneously
        qf1_params = None
        qf2_params = None
        policy_params = None

        for task_futures in train_futures:
            episodes = await task_futures # Assuming train_futures contains the episodes for one task

            # --- Q-Function Update (Inner Loop) ---
            # Compute targets
            with torch.no_grad():
                next_pi = self.policy(episodes.next_observations, params=policy_params)
                next_action, next_log_prob = next_pi.rsample_and_log_prob()
                
                # Reshape for Q-net input
                next_obs_actions = torch.cat([episodes.next_observations, next_action], dim=-1)

                target_q1 = self.target_qf1(next_obs_actions, params=qf1_params)
                target_q2 = self.target_qf2(next_obs_actions, params=qf2_params)
                min_target_q = torch.min(target_q1, target_q2) - self.sac_alpha * next_log_prob.unsqueeze(-1)
                
                # Q-target: R + gamma * (min_Q(s', a') - alpha * log_pi(a'|s'))
                target_v = episodes.rewards + (1.0 - episodes.dones.float()) * self.gamma * min_target_q

            # Predict Q-values
            obs_actions = torch.cat([episodes.observations, episodes.actions], dim=-1)
            q1_pred = self.qf1(obs_actions, params=qf1_params)
            q2_pred = self.qf2(obs_actions, params=qf2_params)
            
            # MSE loss for Q-networks
            qf1_loss = 0.5 * nn.MSELoss(reduction='none')(q1_pred, target_v.detach())
            qf2_loss = 0.5 * nn.MSELoss(reduction='none')(q2_pred, target_v.detach())
            
            qf_loss = weighted_mean(qf1_loss + qf2_loss, episodes.lengths).mean()

            # --- Policy Update (Inner Loop) ---
            pi = self.policy(episodes.observations, params=policy_params)
            new_action, log_prob = pi.rsample_and_log_prob()
            
            # Reshape for Q-net input
            new_obs_actions = torch.cat([episodes.observations, new_action], dim=-1)

            # Predict Q-values using the current Q-network parameters
            q1_new_action = self.qf1(new_obs_actions, params=qf1_params)
            q2_new_action = self.qf2(new_obs_actions, params=qf2_params)
            min_q_new_action = torch.min(q1_new_action, q2_new_action)
            
            # Policy loss: max(Q(s, a) - alpha * log_pi(a|s)) -> min(-Q(s, a) + alpha * log_pi(a|s))
            policy_loss = weighted_mean((self.sac_alpha * log_prob.unsqueeze(-1) - min_q_new_action).mean(dim=0), 
                                        episodes.lengths).mean()
            
            # --- Update Adapted Parameters (Combined Loss) ---
            # Total Inner Loss for gradient flow: only policy loss matters for meta-gradient flow
            # We must update Q and Policy parameters sequentially or jointly.
            # Here we update Q-nets first, then Policy, typical of SAC.

            # 1. Update Q-net parameters
            qf1_params = self.qf1.update_params(qf1_loss, params=qf1_params, step_size=self.fast_lr, first_order=first_order)
            qf2_params = self.qf2.update_params(qf2_loss, params=qf2_params, step_size=self.fast_lr, first_order=first_order)
            
            # 2. Update Policy parameters
            policy_params = self.policy.update_params(policy_loss, params=policy_params, step_size=self.fast_lr, first_order=first_order)

            # Note: Target Q-networks are usually updated via soft-update in the outer loop (self.step)

        # Store the adapted policy and Q-network parameters
        adapted_params_list.append(policy_params)
        qf_params_list.append((qf1_params, qf2_params))
            
        return adapted_params_list, qf_params_list

    async def surrogate_loss(self, train_futures, valid_futures, adapted_params, qf_params):
        """
        Calculates the meta-loss (performance of adapted policy on validation data).
        """
        # The meta-loss is usually the standard RL objective (maximize return)
        # evaluated on the adapted policy using validation data.
        
        valid_episodes = await valid_futures
        policy_params = adapted_params
        
        # Policy evaluation using the adapted policy parameters
        pi = self.policy(valid_episodes.observations, params=policy_params)
        
        # Standard Reinforce Loss (negative weighted return)
        # Note: We can simplify this to just calculating the negative weighted return
        # since the policy gradient is what matters for the outer loop update.
        log_probs = pi.log_prob(valid_episodes.actions)
        
        # Calculate policy loss (negative weighted returns)
        # SAC typically uses the minimum Q-value for advantage, but for the meta-loss
        # to flow through, we must use the standard Reinforce loss structure with
        # advantage estimates if the base sampler/baseline utility provides them.
        # Assuming valid_episodes provides advantages/returns calculated using the baseline
        
        # Use standard RL objective (maximize return) for meta-gradient:
        meta_loss = -weighted_mean(valid_episodes.returns,
                                   lengths=valid_episodes.lengths).mean()
        
        return meta_loss

    def step(self, train_futures, valid_futures):
        num_tasks = len(train_futures)
        logs = {}
        
        # 1. Perform Inner Loop Adaptation for all tasks
        adapted_params_qf_list = self._async_gather([
            self.adapt(train_fut) for train_fut in train_futures
        ])
        
        # Unpack adapted parameters
        adapted_policy_params = [p[0] for p in adapted_params_qf_list]
        adapted_qf_params = [p[1] for p in adapted_params_qf_list]

        # 2. Compute Meta-Loss (Outer Loop Objective)
        # This requires the meta-loss to flow through the adaptation step.
        meta_losses = self._async_gather([
            self.surrogate_loss(train_fut, valid_fut, adapted_policy_params[i], adapted_qf_params[i])
            for i, (train_fut, valid_fut)
            in enumerate(zip(train_futures, valid_futures))
        ])
        
        total_meta_loss = sum(meta_losses) / num_tasks
        logs['meta_loss'] = total_meta_loss.item()

        # 3. Perform Outer Loop (Meta-Parameter) Update
        self.policy_optimizer.zero_grad()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        # Calculate meta-gradient
        total_meta_loss.backward()

        # Update meta-parameters
        self.policy_optimizer.step()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        
        # 4. Soft Update Target Networks (SAC specific)
        soft_update(self.qf1, self.target_qf1, self.tau)
        soft_update(self.qf2, self.target_qf2, self.tau)

        return logs