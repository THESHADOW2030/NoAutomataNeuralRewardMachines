"""
This is a simple wrapper that will include LTL goals to any given environment.
It also progress the formulas as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
    - *get_propositions(...)*: Maps the objects in the environment to a set of
                            propositions that can be referred to in LTL.

Notes about LTLEnv:
    - The episode ends if the LTL goal is progressed to True or False.
    - If the LTL goal becomes True, then an extra +1 reward is given to the agent.
    - If the LTL goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""


import numpy as np
import gym
from gym import spaces

import SymGroundMultiTask.ltl_progression as ltl_progression
from SymGroundMultiTask.ltl_samplers import getLTLSampler, SequenceSampler

from RL.Env.FiniteStateMachine import MooreMachine

from SymGroundMultiTask.envs.gridworld_multitask.Environment import GridWorldEnv_multitask
import cv2

class LTLEnv(gym.Wrapper):

    def __init__(self, env, progression_mode="full", ltl_sampler=None, intrinsic=0.0, formula_old=None, state_type="symbol", use_dfa_state=True):
        """
        LTL environment
        --------------------
        It adds an LTL objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the LTL objective
            - It also automatically progress the formula and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        progression_mode:
            - "full": the agent gets the full, progressed LTL formula as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original LTL formula as part of the observation
        """
        super().__init__(env)
        self.progression_mode = progression_mode
        self.propositions = self.env.get_propositions()
        self.sampler = getLTLSampler(ltl_sampler, self.propositions)

        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.known_progressions = {}
        self.intrinsic = intrinsic

        self.sample_on_reset = True
        self.ltl_original = None
        self.task_id = None


        self.use_dfa_state = use_dfa_state
        self.state_type = state_type
        self.formula_old = formula_old
        self.automaton = MooreMachine(arg1=self.formula_old[0], arg2=self.formula_old[1], arg3=self.formula_old[2], 
                                      reward = "distance", 
                                      dictionary_symbols= self.env.dictionary_symbols)

        self.max_reward = 100 
        print("MAXIMUM REWARD:", self.max_reward)

        if self.state_type == "symbol":
            self.state_space_size = 2

        self.set_for_dict = set(self.automaton.rewards)
        self.list_rew = sorted(self.set_for_dict)
        self.rew_dictionary = {}
        for idx, reward in enumerate(self.list_rew):
            self.rew_dictionary[reward]=idx

        



    def reset(self):

        self.known_progressions = {}
        self.obs = self.env.reset()

        # Defining an LTL goal
        if self.sample_on_reset:
            self.ltl_original, self.task_id = self.sample_ltl_goal()

        self.ltl_goal = self.ltl_original

        # Adding the ltl goal to the observation
        if self.progression_mode == "partial":
            ltl_obs = {
                'features': self.obs,
                'progress_info': self.progress_info(self.ltl_goal)
            }
        else:
            ltl_obs = {
                'features': self.obs,
                'text': self.ltl_goal
            }

        return ltl_obs


    def step(self, action):
        
        int_reward = 0
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the ltl formula
        truth_assignment = self.get_events(self.obs, action, next_obs)
        self.ltl_goal = self.progression(self.ltl_goal, truth_assignment)
        self.obs = next_obs

        # Computing the LTL reward and done signal
        ltl_reward = 0.0
        ltl_done = False
        if self.ltl_goal == 'True':
            ltl_reward = 1.0
            ltl_done = True
        elif self.ltl_goal == 'False':
            ltl_reward = -1.0
            ltl_done = True
        else:
            ltl_reward = int_reward

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            ltl_obs = {
                'features': self.obs,
                'text': self.ltl_goal
            }
        elif self.progression_mode == "none":
            ltl_obs = {
                'features': self.obs,
                'text': self.ltl_original
            }
        elif self.progression_mode == "partial":
            ltl_obs = {
                'features': self.obs,
                'progress_info': self.progress_info(self.ltl_goal)
            }
        else:
            raise NotImplementedError

        reward = original_reward + ltl_reward
        done = env_done or ltl_done
        return ltl_obs, reward, done, info


    def render(self):
        return self.env.render()


    def progression(self, ltl_formula, truth_assignment):
        if (ltl_formula, truth_assignment) not in self.known_progressions:
            result_ltl = ltl_progression.progress_and_clean(ltl_formula, truth_assignment)
            self.known_progressions[(ltl_formula, truth_assignment)] = result_ltl
        return self.known_progressions[(ltl_formula, truth_assignment)]


    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise.
    def progress_info(self, ltl_formula):
        propositions = self.env.get_propositions()
        X = np.zeros(len(self.propositions))
        for i in range(len(propositions)):
            progress_i = self.progression(ltl_formula, propositions[i])
            if progress_i == 'False':
                X[i] = -1.
            elif progress_i != ltl_formula:
                X[i] = 1.
        return X


    def sample_ltl_goal(self):

        # This function must return an LTL formula for the task
        # Format:
        #(
        #    'and',
        #    ('until','True', ('and', 'd', ('until','True',('not','c')))),
        #    ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
        #)
        # NOTE: The propositions must be represented by a char

        formula = self.sampler.sample()

        if isinstance(self.sampler, SequenceSampler):
            def flatten(bla):
                output = []
                for item in bla:
                    output += flatten(item) if isinstance(item, tuple) else [item]
                return output

            length = flatten(formula).count("and") + 1
            self.env.timeout = 25 # 10 * length

        return formula


    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with
        # positive values only(e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()



class NoLTLWrapper(gym.Wrapper):

    def __init__(self, env):
        """
        Removes the LTL formula from an LTLEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space


    def reset(self):
        obs = self.env.reset()
        return obs


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


    def render(self):
        return self.env.render()


    def get_propositions(self):
        return list([])



# a subclass of LTLEnv to distinguish between "real" progrssion and "predicted"
# progression (requires an environment that distinguish between real and
# predicted symbols for an observation)
class LTLGrounderEnv(LTLEnv):

    num_envs = 0


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = LTLGrounderEnv.num_envs
        LTLGrounderEnv.num_envs += 1

        print(self.env.dictionary_symbols)
        print(self.env.num_symbols)
        self.set_for_dict = set(self.automaton.rewards)
        self.list_rew = sorted(self.set_for_dict)
        self.rew_dictionary = {}
        for idx, reward in enumerate(self.list_rew):
            self.rew_dictionary[reward]=idx

        self.task = self.formula_old[2]
        

        

        
    def get_automaton_specs(self):
        num_of_states = self.automaton.num_of_states
        num_of_symbols = len(self.env.dictionary_symbols)
        num_outputs = len(self.list_rew)
        transition_function = self.automaton.transitions
        automaton_rewards = [self.rew_dictionary[rew] for rew in self.automaton.rewards]
        return num_of_states, num_of_symbols, num_outputs, transition_function, automaton_rewards



    def reset(self):

        #ATTENZIONE: QUI CHIAMA IL RESET DEL ENV 

        self.known_progressions = {}
        self.curr_step = 0
        self.obs = self.env.reset()

        self.curr_automaton_state = 0


        if self.state_type == "symbol":
            if self.use_dfa_state:
                observation = np.array(list(
                    self.env.agent_location) + list[self.curr_automaton_state]
                )
            else:
                observation = np.array(list(
                    self.env.agent_location)
                )
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                observation = [np.array(one_hot_dfa_state), self.obs]

            else:
                observation = self.obs
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))


        self.obs = observation
        
        
    


        # sample an LTL goal
        if self.sample_on_reset:
            self.ltl_original, self.task_id = self.sample_ltl_goal()

        # initialize progressed LTL goal
        self.real_ltl_goal = self.ltl_original
        self.pred_ltl_goal = self.ltl_original

        # adding the ltl goal to the observation
        if self.progression_mode == "partial":
            ltl_obs = {
                'features': self.obs,
                'progress_info': self.progress_info(self.pred_ltl_goal),
                'step': self.curr_step,
                'task_id': self.task_id,
                'episode_id': self.env.num_episodes,
                'env_id': self.id
            }
        else:
            ltl_obs = {
                'features': self.obs,
                'text': self.pred_ltl_goal,
                'step': self.curr_step,
                'task_id': self.task_id,
                'episode_id': self.env.num_episodes,
                'env_id': self.id
            }

        reward = 0.0
        info = self._get_info(reward)

        return ltl_obs["features"], reward, info


    def step(self, action):
        
        int_reward = 0.0
        self.curr_step += 1

        # executing the action in the environment. The returned reward is 0 and info is None
        next_obs, env_reward, env_done, info = self.env.step(action)

        
       

        # progressing real ltl formula
        real_label = self.env.get_real_events()
       
        index_label = GridWorldEnv_multitask.symbol_to_index[real_label]
        
        
        self.new_automaton_state = self.automaton.transitions[self.curr_automaton_state][index_label]

        if self.new_automaton_state == self.curr_automaton_state:
            reward = 0
        else:
            reward = self.automaton.rewards[self.new_automaton_state] - self.automaton.rewards[self.curr_automaton_state]
        potential = self.automaton.rewards[self.new_automaton_state]
        self.curr_automaton_state = self.new_automaton_state

        
        
        if self.state_type == "symbol":
            if self.use_dfa_state:
                observation = np.array(list(
                    self.env.agent_location) + list[self.curr_automaton_state]
                )
            else:
                observation = np.array(list(
                    self.env.agent_location)
                )
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                observation = [np.array(one_hot_dfa_state), next_obs]

            else:
                observation = next_obs
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))
        
        next_obs = observation

        
        
        #self.real_ltl_goal = self.progression(self.real_ltl_goal, real_label)

        # progressing pred ltl formula
        #pred_label = self.env.get_events()
        #self.pred_ltl_goal = self.progression(self.pred_ltl_goal, pred_label)

        self.obs = next_obs

        
        

        # computing the new observation and returning the outcome of this action
        # the observation considers the expected formula (unless using 'real')
        if self.progression_mode == "full":
            ltl_obs = {
                'features': self.obs,
                'text': self.pred_ltl_goal,
                'step': self.curr_step,
                'task_id': self.task_id,
                'episode_id': self.env.num_episodes,
                'env_id': self.id
            }
        elif self.progression_mode == "none":
            ltl_obs = {
                'features': self.obs,
                'text': self.ltl_original,
                'step': self.curr_step,
                'task_id': self.task_id,
                'episode_id': self.env.num_episodes,
                'env_id': self.id
            }
        elif self.progression_mode == "partial":
            ltl_obs = {
                'features': self.obs,
                'progress_info': self.progress_info(self.pred_ltl_goal),
                'step': self.curr_step,
                'task_id': self.task_id,
                'episode_id': self.env.num_episodes,
                'env_id': self.id
            }
        elif self.progression_mode == "real":
            ltl_obs = {
                'features': self.obs,
                'text': self.real_ltl_goal,
                'step': self.curr_step,
                'task_id': self.task_id,
                'episode_id': self.env.num_episodes,
                'env_id': self.id
            }
        else:
            raise NotImplementedError

        # the reward considers the real evolution of the formula
        #reward = env_reward + real_ltl_reward

        # the termination checks both real termination or expected one
        done = (potential == 100) or (potential == -100)
        truncated = self.curr_step >= self.env.max_num_steps

        info = self._get_info(potential)
        
        return self.obs, reward, done, truncated, info


    # returns formula and id
    def sample_ltl_goal(self):
        goal_formula = self.sampler.sample()
        goal_id = self.sampler.get_current_id()
        return goal_formula, goal_id
    
    def _get_info(self, reward):
        
        info = self.rew_dictionary[reward]

        return info