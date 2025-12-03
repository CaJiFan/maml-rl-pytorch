import torch
import multiprocessing as mp
import gymnasium as gym
import numpy as np
import asyncio

from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.utils.reinforcement_learning import Episode, BatchEpisodes

# --- Pickleable Environment Factory (Cross-platform compatibility) ---
class EnvFactory:
    def __init__(self, env_name, env_kwargs, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        # Note: Seeding is handled by the VectorEnv reset() calling env.reset(seed=...)
        return env

def make_env(env_name, env_kwargs={}, seed=None):
    return EnvFactory(env_name, env_kwargs, seed)


class SamplerWorker(mp.Process):
    def __init__(self, index, env_name, env_kwargs, batch_size,
                 policy, baseline, task_queue, result_queue, device=None, seed=None):
        super(SamplerWorker, self).__init__()
        self.index = index
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.device = device
        self.seed = seed
        
        # Shared queues passed from the sampler
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        # Recreate environment in worker
        env_fns = [make_env(self.env_name, self.env_kwargs)
                   for _ in range(self.batch_size)]
        
        self.envs = SyncVectorEnv(env_fns)
        self.envs.seed(None if (self.seed is None) 
                       else self.seed + self.index * self.batch_size)

        while True:
            try:
                task = self.task_queue.get()
            except (EOFError, BrokenPipeError):
                break

            if task is None:
                break
            
            self.envs.reset_task(task)
            self.sample(task)
            
    def sample(self, task):
        episodes = self.create_episodes()
        self.result_queue.put(episodes)

    def create_episodes(self):
        episodes = [Episode(None, None, None, None) for _ in range(self.batch_size)]
        
        # Fix 1: Unpack Gymnasium reset (obs, info)
        observations, infos = self.envs.reset()
        
        # Fix 2: Float32 cast
        observations_tensor = torch.from_numpy(observations).float()
        if self.device is not None:
             observations_tensor = observations_tensor.to(self.device)

        batch_ids = list(range(self.batch_size))
        
        while not self.envs.dones.all():
            with torch.no_grad():
                pi = self.policy(observations_tensor)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()

            # Fix 3: Step returns 5 values (obs, rewards, term, trunc, infos)
            # This works because we updated SyncVectorEnv.step() to return this tuple
            next_observations, rewards, terminates, truncates, infos = self.envs.step(actions)
            
            dones = terminates | truncates
            step_batch_ids = infos['batch_ids']

            next_observations_tensor = torch.from_numpy(next_observations).float()
            if self.device is not None:
                next_observations_tensor = next_observations_tensor.to(self.device)

            for i, idx in enumerate(step_batch_ids):
                episodes[idx].append(
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_observations[i],
                    dones[i]
                )

            observations = next_observations
            observations_tensor = next_observations_tensor
            
        for episode in episodes:
            if self.baseline is not None:
                episode.compute_advantages(self.baseline)

        return episodes


class MultiTaskSampler:
    def __init__(self, env_name, env_kwargs, batch_size, policy, baseline=None,
                 env=None, seed=None, num_workers=1, device=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers
        self.device = device

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.workers = [
            SamplerWorker(
                index=i,
                env_name=env_name,
                env_kwargs=env_kwargs,
                batch_size=batch_size,
                policy=policy,
                baseline=baseline,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                device=device,
                seed=seed
            )
            for i in range(num_workers)
        ]

        for worker in self.workers:
            worker.start()

    def sample_tasks(self, num_tasks):
        factory = make_env(self.env_name, self.env_kwargs)
        temp_env = factory()
        tasks = temp_env.unwrapped.sample_tasks(num_tasks)
        temp_env.close()
        return tasks

    def sample_async(self, tasks, num_steps=1, **kwargs):
        """
        Samples batches of episodes for each task. 
        Returns (train_futures, valid_futures).
        """
        # For each task, we need 'num_steps' batches for adaptation (Support)
        # AND 1 batch for validation (Query).
        total_batches_per_task = num_steps + 1
        
        for task in tasks:
            for _ in range(total_batches_per_task):
                self.task_queue.put(task)

        # Collect results
        # Structure: results[task_idx][batch_idx]
        train_results = []
        valid_results = []
        
        async def make_awaitable(res):
            return res

        for _ in range(len(tasks)):
            # Collect 'num_steps' batches for training (Support Set)
            task_train = []
            for _ in range(num_steps):
                episodes = self.result_queue.get()
                batch = BatchEpisodes(episodes, self.batch_size, device=self.device)
                task_train.append(make_awaitable(batch))
            train_results.append(task_train)
            
            # Collect 1 batch for validation (Query Set)
            episodes = self.result_queue.get()
            batch = BatchEpisodes(episodes, self.batch_size, device=self.device)
            valid_results.append(make_awaitable(batch))

        # Returns tuple: (list of lists of futures, list of futures)
        return (train_results, valid_results)

    def close(self):
        for worker in self.workers:
            worker.terminate()
            worker.join()