# from gymnasium.envs.registration import load
# from gymnasium.wrappers import TimeLimit

# from maml_rl.envs.utils.normalized_env import NormalizedActionWrapper

# def mujoco_wrapper(entry_point, **kwargs):
#     normalization_scale = kwargs.pop('normalization_scale', 1.)
#     max_episode_steps = kwargs.pop('max_episode_steps', 200)

#     # Load the environment from its entry point
#     env_cls = load(entry_point)
#     env = env_cls(**kwargs)

#     # Normalization wrapper
#     env = NormalizedActionWrapper(env, scale=normalization_scale)

#     # Time limit
#     env = TimeLimit(env, max_episode_steps=max_episode_steps)

#     return env

import importlib 
from gymnasium.wrappers import TimeLimit

from maml_rl.envs.utils.normalized_env import NormalizedActionWrapper

def mujoco_wrapper(entry_point, **kwargs):
    normalization_scale = kwargs.pop('normalization_scale', 1.)
    # Note: gymnasium.wrappers.TimeLimit automatically sets the max_episode_steps
    # based on the environment spec if available, but since we are wrapping, 
    # using it explicitly like this is fine.
    max_episode_steps = kwargs.pop('max_episode_steps', 200) 

    # --- FIX: Dynamically load the environment class using importlib ---
    # The entry_point is expected to be in the format 'module.submodule:ClassName'
    
    # 1. Split the module path and class name
    mod_name, class_name = entry_point.rsplit(':', 1)
    
    # 2. Import the module
    mod = importlib.import_module(mod_name)
    
    # 3. Get the environment class object
    env_cls = getattr(mod, class_name)
    # --- END FIX ---

    # Instantiate the environment
    env = env_cls(**kwargs)

    # Normalization wrapper
    env = NormalizedActionWrapper(env, scale=normalization_scale)

    # Time limit (using gymnasium.wrappers)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env