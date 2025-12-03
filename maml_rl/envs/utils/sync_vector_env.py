import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv as SyncVectorEnv_
from gymnasium.vector.utils import concatenate, create_empty_array

class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self,
                 env_fns,
                 observation_space=None,
                 action_space=None,
                 **kwargs):
        
        super(SyncVectorEnv, self).__init__(env_fns, **kwargs)
        
        # Initialize internal state
        self._dones = np.zeros((self.num_envs,), dtype=bool)
        
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
        self.reset(seed=seed)
        return seed

    def step(self, actions):
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = len(actions)
        
        rewards = np.zeros((num_actions,), dtype=np.float64)
        terminations = np.zeros((num_actions,), dtype=bool)
        truncations = np.zeros((num_actions,), dtype=bool)
        
        for i, env in enumerate(self.envs):
            # Skip environments that were ALREADY done before this step
            if self._dones[i]:
                continue

            action = actions[i]
            
            # Step the environment
            observation, rewards[i], terminated, truncated, info = env.step(action)
            
            terminations[i] = terminated
            truncations[i] = truncated
            
            # Update internal done state
            self._dones[i] = terminated or truncated
            
            batch_ids.append(i)

            # --- FIX: Always append observation/info if we stepped ---
            # Even if the environment just finished (dones[i] is True), 
            # we need to return the terminal observation so the sampler 
            # can record the final transition (s, a, r, s', done).
            observations_list.append(observation)
            infos.append(info)
            # ---------------------------------------------------------
            
            j += 1

        if observations_list:
            observations = create_empty_array(self.single_observation_space,
                                              n=len(observations_list),
                                              fn=np.zeros)
            concatenate(self.single_observation_space, 
                        observations_list, 
                        observations)
        else:
            observations = np.array([]) 

        return (observations, rewards, terminations, truncations, 
                {'batch_ids': batch_ids, 'infos': infos})