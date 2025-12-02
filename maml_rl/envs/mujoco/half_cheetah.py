import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as HalfCheetahEnv_ 

class HalfCheetahEnv(HalfCheetahEnv_):
    def __init__(self, **kwargs):
        # Initialize the base class first to setup MuJoCo model and data
        super().__init__(**kwargs)
        
        # --- FIX: Update Observation Space ---
        # The base class sets observation_space to (17,). 
        # Since _get_obs adds 3 dimensions (torso COM), we must update it to (20,).
        # We calculate this dynamically to be safe.
        sample_obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=sample_obs.shape, 
            dtype=np.float64 # Standardize on float64 to suppress warnings
        )
        # --- END FIX ---

    def _get_obs(self):
        # --- FIX: Use self.data instead of self.sim.data ---
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float64).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer().render()


class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity."""
    def __init__(self, task={}, low=0.0, high=2.0):
        self._task = task
        self.low = low
        self.high = high

        self._goal_vel = task.get('velocity', 0.0)
        # Pass kwargs if needed, or just init super
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        # --- FIX: Use self.data ---
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        
        # --- FIX: 5-tuple return ---
        terminated = False 
        truncated = False 
                          
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self._task)
        
        return (observation, reward, terminated, truncated, infos)

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction."""
    def __init__(self, task={}):
        self._task = task
        self._goal_dir = task.get('direction', 1)
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        # --- FIX: Use self.data ---
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        
        # --- FIX: 5-tuple return ---
        terminated = False
        truncated = False
        
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self._task)
        
        return (observation, reward, terminated, truncated, infos)

    def sample_tasks(self, num_tasks):
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_dir = task['direction']