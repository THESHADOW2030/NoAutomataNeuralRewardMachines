import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SB3CompatibilityWrapper(gym.Wrapper):
    def __init__(self, env):
        # 1. THE FIX: Assign a dummy observation space to the raw env 
        # so super().__init__() doesn't crash looking for it.
        if not hasattr(env, "observation_space"):
            env.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            
        super().__init__(env)
        
        # 2. Now define the REAL observation space and overwrite it
        if self.env.state_type == "symbolic":
            obs_shape = 3 if self.env.use_dfa_state else 2
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
            )
            
        elif self.env.state_type == "image":
            img_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, 64, 64), dtype=np.float32
            )
            
            if self.env.use_dfa_state:
                num_dfa_states = self.env.automaton.num_of_states
                self.observation_space = spaces.Dict({
                    "dfa_state": spaces.Box(low=0.0, high=1.0, shape=(num_dfa_states,), dtype=np.float32),
                    "image": img_space
                })
            else:
                self.observation_space = img_space

    

    def _format_observation(self, obs):
        """Converts the environment's observation into the exact format SB3 expects."""
        if self.env.state_type == "symbolic":
            return np.array(obs, dtype=np.float32)
            
        elif self.env.state_type == "image":
            if self.env.use_dfa_state:
                # Convert your list [dfa_one_hot, image_tensor] into a dictionary
                return {
                    "dfa_state": np.array(obs[0], dtype=np.float32),
                    "image": np.array(obs[1], dtype=np.float32)
                }
            else:
                # Convert tensor to numpy array
                return np.array(obs, dtype=np.float32)

    def reset(self, **kwargs):
        # 1. THE FIX: Only unpack two values now (obs, info)
        # We also pass **kwargs down in case SB3 sends a seed
        obs, info = self.env.reset(**kwargs)
        
        # Format the info dictionary. SB3 requires string keys.
        if not isinstance(info, dict):
            info = {"reward_info": info}
            
        return self._format_observation(obs), info

    def step(self, action):
        # Your step returns: observation, reward, done, truncated, info
        obs, reward, done, truncated, info = self.env.step(action)
        
        if not isinstance(info, dict):
            info = {"reward_info": info}
            
        return self._format_observation(obs), float(reward), done, truncated, info