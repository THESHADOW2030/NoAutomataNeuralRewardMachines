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

class LTLEnv(gym.Wrapper):

    def __init__(self, env, progression_mode="full", ltl_sampler=None, intrinsic=0.0, formula_old=None, state_type="symbolic", use_dfa_state=False):
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
        self.formula_old = formula_old

        self.dictionary_symbols = ['P', 'L', 'D', 'G', 'E' ]
        print("dictionary_symbols:", self.dictionary_symbols)
        print("formula_old:", self.formula_old)
        self.automaton = MooreMachine(arg1=self.formula_old[0], arg2=self.formula_old[1], arg3=self.formula_old[2], 
                                      reward = "distance", 
                                      dictionary_symbols=self.dictionary_symbols)
        self.max_reward = 100 
        print("MAXIMUM REWARD:", self.max_reward)

        self.set_for_dict = set(self.automaton.rewards)
        self.list_rew = sorted(self.set_for_dict)
        self.rew_dictionary = {}
        for idx, reward in enumerate(self.list_rew):
            self.rew_dictionary[reward]=idx #IMPORTANTE

        self.state_type = state_type

        if state_type == "symbolic":
            self.state_space_size = 2
        elif state_type == "image":
            self.state_space_size = (3, 64,64)


    def reset(self):

        '''
        TUTTO IL RESET
        '''
        self.curr_automaton_state = 0
        self.curr_step = 0
        self._agent_location = np.array([0, 0])

        #if self.render_mode == "human":
        #    self._render_frame()
        if self.state_type == "symbolic":
            if self.use_dfa_state:
                observation = np.array(list(self._agent_location) + [self.curr_automaton_state])
            else:
                observation = np.array(list(self._agent_location))
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                #print("one_hot_dfa_state: ", one_hot_dfa_state)
                observation = [np.array(one_hot_dfa_state), self.image_locations[self._agent_location[0], self._agent_location[1]]] #1 FULL Img, 0 Just the square the robot is in
            else:
                observation = self.image_locations[self._agent_location[0], self._agent_location[1]]
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))

        reward = 0
        info = self.rew_dictionary[reward]
        
        return observation, reward, info



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
        obs, reward, info = self.env.reset()
        return obs, reward, info


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


    def reset(self):

        obs, reward, info = self.env.reset()

        return obs, reward, info


    def step(self, action):     #IMPORTANTE

        reward = -1
        self.curr_step += 1
        done = False

        # MOVEMENT
        if action == 0:
            direction = np.array([0, 1])
        elif action == 1:
            direction = np.array([1, 0])
        elif action == 2:
            direction = np.array([0, -1])
        elif action == 3:
            direction = np.array([-1, 0])

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        sym = self._current_symbol()
        #print("symbol:", sym)
        self.new_automaton_state = self.automaton.transitions[self.curr_automaton_state][sym]
        #print("state:", self.curr_automaton_state)
        #print(self.automaton.acceptance)

        #if self.automaton.acceptance[self.curr_automaton_state]:
        #    reward = 100
        #    done = True
        if self.new_automaton_state == self.curr_automaton_state:
            reward = 0
        else:
            reward = self.automaton.rewards[self.new_automaton_state] - self.automaton.rewards[self.curr_automaton_state]
        potential = self.automaton.rewards[self.new_automaton_state]
        self.curr_automaton_state = self.new_automaton_state

        #if self.render_mode == "human":
        #    self._render_frame()

        if self.state_type == "symbolic":
            if self.use_dfa_state:
                observation = np.array(list(self._agent_location) + [self.curr_automaton_state])
            else:
                observation = np.array(list(self._agent_location))
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                #print("one_hot_dfa_state: ", one_hot_dfa_state)
                observation = [np.array(one_hot_dfa_state), self.image_locations[self._agent_location[0], self._agent_location[1]]]
            else:
                observation = self.image_locations[self._agent_location[0], self._agent_location[1]]

        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))
            
        #          success            failure                  timeout
        done = (potential == 100) or (potential == -100)
        truncated = (self.curr_step >= self.max_num_steps)

        info = self._get_info(potential)

        return observation, reward, done, truncated, info#, sym

    def get_automaton_specs(self):
        num_of_states = self.automaton.num_of_states
        num_of_symbols = len(self.dictionary_symbols)
        num_outputs = len(self.list_rew)
        transition_function = self.automaton.transitions
        automaton_rewards = [self.rew_dictionary[rew] for rew in self.automaton.rewards]
        return num_of_states, num_of_symbols, num_outputs, transition_function, automaton_rewards



    # returns formula and id
    def sample_ltl_goal(self):
        goal_formula = self.sampler.sample()
        goal_id = self.sampler.get_current_id()
        return goal_formula, goal_id