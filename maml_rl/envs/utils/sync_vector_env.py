import numpy as np
import gymnasium as gym # Added gymnasium import if needed later
from gymnasium.vector import SyncVectorEnv as SyncVectorEnv_
from gymnasium.vector.utils import concatenate, create_empty_array


class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self,
                 env_fns,
                 observation_space=None, # Accepts argument from caller
                 action_space=None,      # Accepts argument from caller
                 **kwargs):

        super(SyncVectorEnv, self).__init__(env_fns, **kwargs) 
        
        for env in self.envs:
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('The environment provided is not a '
                                 'meta-learning environment. It does not have '
                                 'the method `reset_task` implemented.')

    @property
    def dones(self):
        return self._dones

    def reset_task(self, task):
        for env in self.envs:
            env.unwrapped.reset_task(task)

    def seed(self, seed=None):
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        
        # We call reset() here to apply the seed. Note that reset() returns (obs, info).
        # We ignore the returns as the caller (.seed()) expects no return or a list of seeds.
        self.reset(seed=seed)

    def step_wait(self):
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = len(self._actions)
        rewards = np.zeros((num_actions,), dtype=np.float_)
        for i, env in enumerate(self.envs):
            if self._dones[i]:
                continue

            action = self._actions[j]
            # observation, rewards[j], self._dones[i], info = env.step(action) # <-- Old gym 4-tuple format
            
            # --- FIX: Update step signature to handle gymnasium 5-tuple ---
            observation, rewards[j], terminated, truncated, info = env.step(action)
            self._dones[i] = terminated or truncated
            # --- END FIX ---
            
            batch_ids.append(i)

            if not self._dones[i]:
                observations_list.append(observation)
                infos.append(info)
            j += 1
        assert num_actions == j

        if observations_list:
            # Note: This logic for concatenation and array creation looks correct for NumPy/vector environments
            observations = create_empty_array(self.single_observation_space,
                                              n=len(observations_list),
                                              fn=np.zeros)
            concatenate(observations_list,
                        observations,
                        self.single_observation_space)
        else:
            observations = None

        # The old gym vector env step_wait returned 4 items: (obs, rewards, dones, infos dict)
        # Assuming old 4-item return is expected by caller:
        return (observations, rewards, np.copy(self._dones),
                {'batch_ids': batch_ids, 'infos': infos})