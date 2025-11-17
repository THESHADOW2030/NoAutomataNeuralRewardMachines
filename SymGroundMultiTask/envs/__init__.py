from gym.envs.registration import register

from SymGroundMultiTask.envs.gym_letters.letter_env import LetterEnv
from SymGroundMultiTask.envs.simple_ltl.simple_ltl_env import SimpleLTLEnv
from SymGroundMultiTask.envs.minigrid.minigrid_env import MinigridEnv
from SymGroundMultiTask.envs.gridworld_multitask.Environment import GridWorldEnv_LTL2Action
from SymGroundMultiTask.envs.safety.zones_env import ZonesEnv

__all__ = ["LetterEnv", "SimpleLTLEnv", "MinigridEnv", "GridWorldEnv_LTL2Action", "ZonesEnv"]


### GridWorld multi-task Envs (5 symbols)
register(
    id='GridWorld-v0',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_Base'
)

register(
    id='GridWorld-fixed-v0',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_Base_FixedMap'
)

register(
    id='GridWorld-v1',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_AgentCentric'
)

register(
    id='GridWorld-fixed-v1',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_AgentCentric_FixedMap'
)

register(
    id='GridWorld-v2',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_NoWrapAround'
)

register(
    id='GridWorld-fixed-v2',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_NoWrapAround_FixedMap'
)



### GridWorld multi-task Envs (12 symbols)
register(
    id='GridWorld-12-v0',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_12_Base'
)

register(
    id='GridWorld-12-fixed-v0',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_12_Base_FixedMap'
)

register(
    id='GridWorld-12-v1',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_12_AgentCentric'
)

register(
    id='GridWorld-12-fixed-v1',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_12_AgentCentric_FixedMap'
)

register(
    id='GridWorld-12-v2',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_12_NoWrapAround'
)

register(
    id='GridWorld-12-fixed-v2',
    entry_point='SymGroundMultiTask.envs.gridworld_multitask.Environment:GridWorldEnv_12_NoWrapAround_FixedMap'
)



### Simple LTL Envs
register(
    id='Simple-LTL-Env-v0',
    entry_point='SymGroundMultiTask.envs.simple_ltl.simple_ltl_env:SimpleLTLEnvDefault'
)

register(
    id='Simple-LTL-Env-5L-v0',
    entry_point='SymGroundMultiTask.envs.simple_ltl.simple_ltl_env:SimpleLTLEnv5Letters'
)


### Letter Envs
register(
    id='Letter-4x4-v0',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnv4x4'
)

register(
    id='Letter-4x4-v1',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvFixedMap4x4'
)

register(
    id='Letter-5x5-v0',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnv5x5'
)

register(
    id='Letter-5x5-v1',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvFixedMap5x5'
)

register(
    id='Letter-5x5-v2',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvAgentCentric5x5'
)

register(
    id='Letter-5x5-v3',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMap5x5'
)

register(
    id='Letter-5x5-v4',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvShortAgentCentric5x5'
)

register(
    id='Letter-5x5-v5',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvShortAgentCentricFixedMap5x5'
)

register(
    id='Letter-7x7-v0',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnv7x7'
)

register(
    id='Letter-7x7-v1',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvFixedMap7x7'
)

register(
    id='Letter-7x7-v2',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvAgentCentric7x7'
)

register(
    id='Letter-7x7-v3',
    entry_point='SymGroundMultiTask.envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMap7x7'
)



### Minigrid Envs
register(
    id='Adversarial-v0',
    entry_point='SymGroundMultiTask.envs.minigrid.minigrid_env:AdversarialMinigridEnv'
)



### Safety Envs
register(
    id='Zones-1-v0',
    entry_point='SymGroundMultiTask.envs.safety.zones_env:ZonesEnv1')

register(
    id='Zones-1-v1',
    entry_point='SymGroundMultiTask.envs.safety.zones_env:ZonesEnv1Fixed')

register(
    id='Zones-5-v0',
    entry_point='SymGroundMultiTask.envs.safety.zones_env:ZonesEnv5')

register(
    id='Zones-5-v1',
    entry_point='SymGroundMultiTask.envs.safety.zones_env:ZonesEnv5Fixed')