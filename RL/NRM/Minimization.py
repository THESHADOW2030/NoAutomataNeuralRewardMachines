
import pickle
from operator import indexOf

import torch

from .utils import transacc2pythomata

from .UnremovableReasoningShurtcuts import find_reasoning_shortcuts, substitute_map


class MinimizableMooreMachine():
    def __init__(self, pythomata_dfa):
        self.pydfa = pythomata_dfa
        self.init_state = self.pydfa._initial_state
        trans = self.pydfa._transition_function
        self.transitions = {}
        for q in trans:
            self.transitions[q] = {}
            for sym in trans[q]:
                self.transitions[q][int(sym)] = trans[q][sym]
        self.num_of_states = len(self.transitions.keys())
        self.acceptance = [(q in self.pydfa._accepting_states) for q in range(self.num_of_states)]
        self.alphabet = list(self.transitions[0].keys())
        self.num_of_symbols = len(self.alphabet)
        self.calculate_absorbing_states()
        self.assign_rewards(criterion= "acceptance")

    def accepts(self, trace):
        if trace == []:
            return self.acceptance[self.init_state]
        return self.accepts_from_state(self.init_state, trace)

    def accepts_from_state(self, state, trace):
        assert trace != []

        a = trace[0]
        next_state = self.transitions[state][a]

        if len(trace) == 1:
            return self.acceptance[next_state]

        return self.accepts_from_state(next_state, trace[1:])

    def process_trace(self, trace):
        return self.process_trace_from_state(trace, self.init_state)

    def process_trace_from_state(self, trace, state):
        a = trace[0]
        next_state = self.transitions[state][a]

        if len(trace) == 1:
            return next_state, self.rewards[next_state]

        return self.process_trace_from_state(trace[1:], next_state)

    def calculate_absorbing_states(self):
        self.absorbing_states = []
        for q in range(self.num_of_states):
            absorbing = True
            for s in self.transitions[q].keys():
                absorbing = absorbing & (self.transitions[q][s] == q)
            if absorbing:
                self.absorbing_states.append(q)

    def assign_rewards(self, criterion = "acceptance"):
        if criterion == "distance":
            self.rewards = [100 for _ in range(self.num_of_states)]
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 0
            # print(self.rewards)
            old_rew = self.rewards.copy()
            termination = False
            while not termination:
                termination = True
                for s in range(self.num_of_states):
                    if not self.acceptance[s]:
                        next = [self.rewards[self.transitions[s][sym]] for sym in self.alphabet if
                                self.transitions[s][sym] != s]
                        if len(next) > 0:
                            self.rewards[s] = 1 + min(next)

                termination = (str(self.rewards) == str(old_rew))
                old_rew = self.rewards.copy()

            for i in range(len(self.rewards)):
                self.rewards[i] *= -1
            minimum = min([r for r in self.rewards if r != -100])
            for i, r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = (r - minimum)

            maximum = max(self.rewards)
            # max : 100 = rew : x
            # x = 100 * rew / max
            for i, r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = 100 * r / maximum
        elif criterion == "acceptance":
            self.rewards = [int(a) for a in self.acceptance]
        #print("REWARDS:", self.rewards)

    def apply_symbols_map(self, map):
        new_transitions = {}
        for q in range(self.num_of_states):
            new_transitions[q] = {}
            for p in range(self.num_of_symbols):
                new_transitions[q][map[p]] = self.transitions[q][p]

        self.transitions = new_transitions
        print("old alphabet: ", self.alphabet)
        self.alphabet = substitute_map(self.alphabet, map)
        print("new alphabet: ", self.alphabet)

    def return_minimized_pydfa(self):
        #eliminate useless symbols
        deleted_symbols = self.eliminate_useless_symbols()

        #eliminate useless states
        pyautomaton = transacc2pythomata(self.transitions, self.acceptance, set(self.alphabet), initial_state=self.init_state)
        pyautomaton = pyautomaton.reachable()
        pyautomaton = pyautomaton.minimize()

        #delete_symbols and alphabet (which is traformed according to the urs found) are needed to construct the tranformations to apply to
        # the classifier
        return pyautomaton, deleted_symbols, self.alphabet

    def eliminate_useless_symbols(self):

        deleted_syms = []
        for sym in set(self.alphabet):
            #if a symbol never changes the state is useless
            change_state = False
            for q in range(self.num_of_states):
                if self.transitions[q][sym] != q:
                    change_state = True
                    break
            if not change_state:
                deleted_syms.append(sym)
                for q in range(self.num_of_states):
                    del self.transitions[q][sym]
        return deleted_syms

def minimize_dfa_symbols_and_states(pythomata_dfa):
    #print(pythomata_dfa.__dict__)
    minMM = MinimizableMooreMachine(pythomata_dfa)
    #print(minMM.__dict__)

    urs, _ = find_reasoning_shortcuts(minMM, mode= "no_renaming")
    urs = list(urs)
    len_urs = []

    if len_urs == []:
            return minMM.return_minimized_pydfa()

    min_rs = urs[indexOf(len_urs, min(len_urs))]
    print("minimal urs: ", min_rs)

    #apply minimal URS to dfa
    minMM.apply_symbols_map(min_rs)

    return  minMM.return_minimized_pydfa()


def create_mask_classifier(deleted_syms, transformed_alphabet):
    #map old alphabet to new alphabet
    old_size = len(transformed_alphabet)
    new_size = len(set(transformed_alphabet))

    transformation_matrix = torch.zeros((old_size, new_size))

    for i, new_sym in enumerate(transformed_alphabet):
        transformation_matrix[i, new_sym] = 1

    delete_sym_filter = torch.ones(new_size)

    for sym in deleted_syms:
        delete_sym_filter[sym] = 0

    return transformation_matrix, delete_sym_filter

