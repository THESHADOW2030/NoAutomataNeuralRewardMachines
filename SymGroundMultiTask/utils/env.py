"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
from SymGroundMultiTask.ltl_wrappers import LTLEnv, NoLTLWrapper, LTLGrounderEnv



def make_env(env_key, progression_mode, ltl_sampler, seed=None, intrinsic=0, noLTL=False, 
    state_type='image', grounder=None, obs_size=None):

    kwargs = {} 
    if "GridWorld" in env_key:
        kwargs = {
            "state_type": state_type,
            "grounder": grounder,
            "obs_size": obs_size
        }

    env = gym.make(env_key, **kwargs)
    env.seed(seed)

    if (noLTL):
        wrapper = NoLTLWrapper(env)

    elif "GridWorld" in env_key:
        
        wrapper = LTLGrounderEnv(
            env=env,
            progression_mode=progression_mode,
            ltl_sampler=ltl_sampler,
            intrinsic=intrinsic
        )

    else:
        wrapper = LTLEnv(
            env=env,
            progression_mode=progression_mode,
            ltl_sampler=ltl_sampler,
            intrinsic=intrinsic
        )

    return wrapper