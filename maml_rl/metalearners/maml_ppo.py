# import torch
# import torch.nn.functional as F

# from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
# from torch.distributions.kl import kl_divergence

# from maml_rl.metalearners.base import GradientBasedMetaLearner
# from maml_rl.utils.reinforcement_learning import reinforce_loss
# from maml_rl.utils.torch_utils import weighted_mean, detach_distribution, to_numpy


# class MAMLPPO(GradientBasedMetaLearner):
#     """
#     MAML with PPO (Proximal Policy Optimization) as the outer-loop optimizer.

#     Parameters:
#     ----------
#     policy : torch.nn.Module
#         Policy network returning a distribution over actions.
#     fast_lr : float
#         Learning rate for the inner loop adaptation.
#     clip_eps : float
#         PPO clipping epsilon.
#     ent_coef : float
#         Entropy bonus coefficient.
#     first_order : bool
#         If True, uses first-order approximation.
#     device : str
#         Device identifier (e.g., 'cpu' or 'cuda').
#     """
#     def __init__(
#             self, 
#             policy,
#             fast_lr=0.1, 
#             clip_eps=0.2, 
#             ent_coef=0.0, 
#             first_order=False, 
#             device='cpu',
#             outer_lr=1e-3
#         ):
#         super().__init__(policy, device=device)
#         self.fast_lr = fast_lr
#         self.clip_eps = clip_eps
#         self.ent_coef = ent_coef
#         self.first_order = first_order
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=outer_lr)

#     async def adapt(self, train_futures, first_order=None):
#         if first_order is None:
#             first_order = self.first_order

#         params = None
#         for futures in train_futures:
#             inner_loss = reinforce_loss(self.policy, await futures, params=params)
#             params = self.policy.update_params(inner_loss, params=params, step_size=self.fast_lr, first_order=first_order)

#         return params

#     async def surrogate_loss(self, train_futures, valid_futures):
#         params = await self.adapt(train_futures)
#         valid_episodes = await valid_futures

#         pi = self.policy(valid_episodes.observations, params=params)
#         with torch.no_grad():
#             old_pi = detach_distribution(self.policy(valid_episodes.observations))

#         log_ratio = pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions)
#         ratio = torch.exp(log_ratio)

#         clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
#         surrogate = torch.min(ratio * valid_episodes.advantages, clipped_ratio * valid_episodes.advantages)
#         loss = -weighted_mean(surrogate, lengths=valid_episodes.lengths)

#         ent = weighted_mean(pi.entropy(), lengths=valid_episodes.lengths)
#         total_loss = loss - self.ent_coef * ent
#         kl = weighted_mean(kl_divergence(pi, old_pi), lengths=valid_episodes.lengths)

#         return total_loss.mean(), kl.mean()

#     def step(self, train_futures, valid_futures):
#         num_tasks = len(train_futures[0])
#         logs = {}

#         losses, kls = self._async_gather([
#             self.surrogate_loss(train, valid)
#             for (train, valid) in zip(zip(*train_futures), valid_futures)
#         ])

#         total_loss = sum(losses) / num_tasks
#         self.policy.zero_grad()
#         total_loss.backward()
#         self.optimizer.step()

#         logs['loss'] = to_numpy(losses)
#         logs['kl'] = to_numpy(kls)

#         return logs

import torch
import torch.nn as nn

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml_rl.samplers import MultiTaskSampler # Assuming this exists
from maml_rl.metalearners.base import GradientBasedMetaLearner # Assuming this exists
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters) # Assuming these exist
from maml_rl.utils.optimization import conjugate_gradient # TRPO specific, not needed for PPO outer
from maml_rl.utils.reinforcement_learning import reinforce_loss # Inner loop loss

class MAML_PPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on Proximal Policy
    Optimization (PPO, [2]).

    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied during
        inner loop updates.

    outer_lr : float
        Step-size for the outer loop optimization (meta-optimization).

    ppo_clip_param : float
        Clipping parameter for the PPO surrogate objective.

    ppo_epochs : int
        Number of gradient descent steps to perform on the outer parameters
        per meta-iteration.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Chen, X. (2017).
           Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347 (https://arxiv.org/abs/1707.06347)
    """
    def __init__(self,
                 policy,
                 fast_lr=0.1, # Typically smaller than TRPO fast_lr
                 first_order=False,
                 outer_lr=1e-3, # Standard learning rate for outer optimizer
                 ppo_clip_param=0.2,
                 ppo_epochs=3, # Multiple epochs for outer loop
                 device='cpu'):
        super(MAML_PPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order
        self.outer_lr = outer_lr
        self.ppo_clip_param = ppo_clip_param
        self.ppo_epochs = ppo_epochs

        # Outer loop optimizer (Adam is common for PPO)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.outer_lr)

    async def adapt(self, train_futures, first_order=None):
        """
        Performs the inner loop adaptation for each task.

        Parameters
        ----------
        train_futures : list of Futures
            A list of futures, where each future resolves to training data
            (episodes) for a specific task.

        first_order : bool, optional
            Override the default `first_order` setting for this adaptation step.

        Returns
        -------
        list of dict
            A list of task-specific parameters after adaptation, one for each task.
        """
        if first_order is None:
            first_order = self.first_order

        adapted_params_list = []
        # Loop over tasks
        for task_futures in train_futures:
            params = None # Start with initial meta-parameters for each task
            # Loop over the number of steps of adaptation
            inner_loss = reinforce_loss(self.policy,
                                        await task_futures,
                                        params=params) # Use await directly on the single future
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
            adapted_params_list.append(params)

        return adapted_params_list

    async def compute_ppo_loss(self, valid_episodes, adapted_params, old_pi):
        """
        Computes the PPO loss for a single task using the adapted parameters.

        Parameters
        ----------
        valid_episodes : Episode instance
            Validation data (episodes) for a specific task.
        adapted_params : dict
            The task-specific parameters after inner loop adaptation.
        old_pi : torch.distributions.Distribution
            The policy distribution from the previous outer iteration, used
            as the reference for the ratio calculation.

        Returns
        -------
        torch.Tensor
            The computed PPO loss for the task.
        torch.Tensor
            The KL divergence between the current and old policy.
        """
        # Evaluate the policy with adapted parameters on validation data
        pi = self.policy(valid_episodes.observations, params=adapted_params)

        log_ratio = (pi.log_prob(valid_episodes.actions)
                     - old_pi.log_prob(valid_episodes.actions))
        ratio = torch.exp(log_ratio)

        # PPO clipped surrogate objective
        surr1 = ratio * valid_episodes.advantages
        surr2 = torch.clamp(ratio,
                            1.0 - self.ppo_clip_param,
                            1.0 + self.ppo_clip_param) * valid_episodes.advantages
        # We minimize negative advantage, so max becomes min
        loss = -torch.min(surr1, surr2)

        # Optional: Add an entropy bonus or value function loss if applicable
        # entropy = pi.entropy().mean()
        # loss = loss - self.entropy_coef * entropy

        task_loss = weighted_mean(loss, lengths=valid_episodes.lengths)
        task_kl = weighted_mean(kl_divergence(pi, old_pi), lengths=valid_episodes.lengths)

        return task_loss, task_kl


    def step(self, train_futures, valid_futures):
        """
        Performs the outer loop optimization step using PPO.

        Parameters
        ----------
        train_futures : list of list of Futures
             A list of lists of futures, structured as
            [[task1_step1_future, task1_step2_future, ...],
             [task2_step1_future, task2_step2_future, ...], ...]
             For single-step adaptation, it's [[task1_future], [task2_future], ...].
        valid_futures : list of Futures
            A list of futures, where each future resolves to validation data
            (episodes) for a specific task.

        Returns
        -------
        dict
            A dictionary containing logs from the optimization step (e.g.,
            average loss and KL before/after the update).
        """
        num_tasks = len(train_futures) # Assuming train_futures is structured as [task1_futures, task2_futures, ...]
        logs = {}

        # 1. Perform inner loop adaptation for each task
        # train_futures structure needs to match adapt() expected input.
        # Assuming train_futures is [[task1_step1_future], [task2_step1_future], ...]
        # and adapt expects [task1_step1_future, task2_step1_future, ...]
        # Let's adjust train_futures for adapt call. Assuming single inner step.
        train_futures_for_adapt = [tf[0] for tf in train_futures] if train_futures and isinstance(train_futures[0], list) else train_futures

        # Gather adapted parameters for all tasks
        adapted_params_list = self._async_gather(self.adapt(train_futures_for_adapt, first_order=False)) # Use second-order for outer loop gradient if possible

        # Gather validation data for all tasks
        valid_episodes_list = self._async_gather(valid_futures)

        # 2. Compute PPO loss and KL divergence for each task using adapted parameters
        # and the current meta-policy as the 'old' policy
        old_pis = self._async_gather([
            detach_distribution(self.policy(ve.observations, params=ap))
            for ve, ap in zip(valid_episodes_list, adapted_params_list)
        ])

        # Compute initial losses and KLs before outer loop update
        old_losses_kls = self._async_gather([
             self.compute_ppo_loss(ve, ap, old_pi)
             for ve, ap, old_pi in zip(valid_episodes_list, adapted_params_list, old_pis)
        ])
        old_losses, old_kls = zip(*old_losses_kls)

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        # 3. Perform PPO outer loop updates
        self.optimizer.zero_grad()
        total_outer_loss = 0

        # We will perform multiple epochs of gradient descent on the outer loss
        # The gradients for the outer loss must flow through the inner adaptation.
        # We compute the gradient of the sum of task losses w.r.t. the initial parameters theta
        outer_loss_for_grad = sum(old_losses) / num_tasks
        outer_loss_for_grad.backward()

        # Perform optimization step
        self.optimizer.step()

        # (Optional) Re-evaluate losses and KLs after outer loop update for logging
        # Note: This requires re-running adaptation with the updated meta-parameters
        # which can be computationally expensive. For simplicity in this example,
        # we omit re-evaluation and just log the loss before the step.
        # A more complete implementation might evaluate after the full PPO epochs.

        # Example re-evaluation (commented out for basic version)
        # with torch.no_grad(): # No gradient needed for logging
        #     adapted_params_list_after = self._async_gather(self.adapt(train_futures_for_adapt, first_order=False))
        #     losses_kls_after = self._async_gather([
        #         self.compute_ppo_loss(ve, ap_after, old_pi)
        #         for ve, ap_after, old_pi in zip(valid_episodes_list, adapted_params_list_after, old_pis)
        #     ])
        #     losses_after, kls_after = zip(*losses_kls_after)
        #     logs['loss_after'] = to_numpy(losses_after)
        #     logs['kl_after'] = to_numpy(kls_after)


        return logs