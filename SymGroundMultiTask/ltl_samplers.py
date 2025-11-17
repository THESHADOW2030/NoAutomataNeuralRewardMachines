"""
This class is responsible for sampling LTL formulas typically from
given template(s).

@ propositions: The set of propositions to be used in the sampled
                formula at random.
"""

import random
import os
import pickle
import yaml
import numpy as np



class LTLSampler():

    def __init__(self, propositions):
        self.propositions = propositions
        self.has_automata = False


    def sample(self):
        raise NotImplementedError


    def get_current_id(self):
        return -1



# Samples from one of the other samplers at random. The other samplers are sampled by their default args.
class SuperSampler(LTLSampler):

    def __init__(self, propositions):
        super().__init__(propositions)
        self.reg_samplers = getRegisteredSamplers(self.propositions)


    def sample(self):
        return random.choice(self.reg_samplers).sample()



# This class samples formulas of form (or, op_1, op_2), where op_1 and 2 can be either specified as samplers_ids
# or by default they will be sampled at random via SuperSampler.
class OrSampler(LTLSampler):

    def __init__(self, propositions, sampler_ids = ["SuperSampler"]*2):
        super().__init__(propositions)
        self.sampler_ids = sampler_ids


    def sample(self):
        return ('or', getLTLSampler(self.sampler_ids[0], self.propositions).sample(),
                        getLTLSampler(self.sampler_ids[1], self.propositions).sample())



# This class generates random LTL formulas using the following template:
#   ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
# where p1, p2, p3, and p4 are randomly sampled propositions
class DefaultSampler(LTLSampler):

    def sample(self):
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))



# This class generates random conjunctions of Until-Tasks.
# Each until tasks has *n* levels, where each level consists
# of avoiding a proposition until reaching another proposition.
#   E.g.,
#      Level 1: ('until',('not','a'),'b')
#      Level 2: ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
#      etc...
# The number of until-tasks, their levels, and their propositions are randomly sampled.
# This code is a generalization of the DefaultSampler---which is equivalent to UntilTaskSampler(propositions, 2, 2, 1, 1)
class UntilTaskSampler(LTLSampler):

    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1, max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"


    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
        return ltl



# This class generates random LTL formulas that form a sequence of actions.
# @ min_len, max_len: min/max length of the random sequence to generate.
class SequenceSampler(LTLSampler):

    def __init__(self, propositions, min_len=2, max_len=4):
        super().__init__(propositions)
        self.min_len = int(min_len)
        self.max_len = int(max_len)


    def sample(self):
        length = random.randint(self.min_len, self.max_len)
        seq = ""

        while len(seq) < length:
            c = random.choice(self.propositions)
            if len(seq) == 0 or seq[-1] != c:
                seq += c

        ret = self._get_sequence(seq)

        return ret


    def _get_sequence(self, seq):
        if len(seq) == 1:
            return ('eventually',seq)
        return ('eventually',('and', seq[0], self._get_sequence(seq[1:])))



# This generates several sequence tasks which can be accomplished in parallel. 
# e.g. in (eventually (a and eventually c)) and (eventually b)
# the two sequence tasks are "a->c" and "b".
class EventuallySampler(LTLSampler):

    def __init__(self, propositions, min_levels=1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))


    def sample(self):

        conjs = random.randint(*self.conjunctions)

        ltl = None
        for i in range(conjs):
            task = self.sample_sequence()
            if ltl is None:
                ltl = task
            else:
                ltl = ('and',task,ltl)

        return ltl


    def sample_sequence(self):
        length = random.randint(*self.levels)
        seq = []
        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [p for p in self.propositions if p not in last]
            if random.random() < 0.25:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)
            seq.append(c)
            last = c
        ret = self._get_sequence(seq)
        return ret


    def _get_sequence(self, seq):
        term = seq[0][0] if len(seq[0]) == 1 else ('or', seq[0][0], seq[0][1])
        if len(seq) == 1:
            return ('eventually',term)
        return ('eventually',('and', term, self._get_sequence(seq[1:])))



# This generates several sequence tasks which can be accomplished in parallel with a global avoidance.
# These tasks are not co-safe
class TrueGlobalAvoidanceSampler(LTLSampler):

    def __init__(self, propositions, min_levels=1, max_levels=4, min_conjunctions=1, max_conjunctions=3, min_avoid=1, max_avoid=2):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))
        self.avoids = (int(min_avoid), int(max_avoid))


    def sample(self):

        n_avoids = random.randint(*self.avoids)
        avoids = random.sample(self.propositions, n_avoids)
        remaining = [item for item in self.propositions if item not in avoids]

        conjs = random.randint(*self.conjunctions)
        ltl = None
        for i in range(conjs):
            task = self.sample_sequence(remaining)
            if ltl is None:
                ltl = task
            else:
                ltl = ('and', task, ltl)

        for p in avoids:
            avoidance = ('always', ('not', p))
            if ltl is None:
                ltl = avoidance
            else:
                ltl = ('and', avoidance, ltl)

        return ltl


    def sample_sequence(self, propositions):
        length = random.randint(*self.levels)
        seq = []
        last = []
        while len(seq) < length:
            population = [p for p in propositions if p not in last]
            if random.random() < 0.25 and len(population) >= 2:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)
            seq.append(c)
            last = c
        ret = self._get_sequence(seq)
        return ret


    def _get_sequence(self, seq):
        term = seq[0][0] if len(seq[0]) == 1 else ('or', seq[0][0], seq[0][1])
        if len(seq) == 1:
            return ('eventually',term)
        return ('eventually',('and', term, self._get_sequence(seq[1:])))



# This generates several sequence tasks which can be accomplished in parallel with a global avoidance.
# (special case of the until task sampler with the same avoidances at each step)
# e.g. ('until', ('and',('not','a'), ('not','c')), ('and', 'b', ('until',('and',('not','a'), ('not','c')),'d')))
# the sequence is b->d without passing for a and c
# These tasks are co-safe
class GlobalAvoidanceSampler(LTLSampler):

    def __init__(self, propositions, min_levels=1, max_levels=4, min_conjunctions=1, max_conjunctions=3, min_avoid=1, max_avoid=2):
        super().__init__(propositions)
        assert(len(propositions) >= int(max_avoid) + 2)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))
        self.avoids = (int(min_avoid), int(max_avoid))


    def sample(self):

        n_avoids = random.randint(*self.avoids)
        avoids = random.sample(self.propositions, n_avoids)
        remaining = [item for item in self.propositions if item not in avoids]

        avoidance = self._get_avoidance(avoids)

        conjs = random.randint(*self.conjunctions)
        ltl = None
        for i in range(conjs):
            task = self.sample_sequence(remaining, avoidance)
            if ltl is None:
                ltl = task
            else:
                ltl = ('and', task, ltl)

        return ltl


    def sample_sequence(self, propositions, avoidance):
        length = random.randint(*self.levels)
        seq = []
        last = []
        while len(seq) < length:
            population = [p for p in propositions if p not in last]
            c = random.sample(population, 1)
            seq.append(c)
            last = c
        ret = self._get_sequence(seq, avoidance)
        return ret


    def _get_sequence(self, seq, avoidance):
        term = seq[0][0]
        if len(seq) == 1:
            return ('until', avoidance, term)
        return ('until', avoidance, ('and', term, self._get_sequence(seq[1:], avoidance)))


    def _get_avoidance(self, avoids):
        term = avoids[0]
        if len(avoids) == 1:
            return ('not', term)
        return ('and', ('not', term), self._get_avoidance(avoids[1:]))



class AdversarialEnvSampler(LTLSampler):

    def sample(self):
        p = random.randint(0,1)
        if p == 0:
            return ('eventually', ('and', 'a', ('eventually', 'b')))
        else:
            return ('eventually', ('and', 'a', ('eventually', 'c')))



def getRegisteredSamplers(propositions):
    return [SequenceSampler(propositions),
            UntilTaskSampler(propositions),
            DefaultSampler(propositions),
            EventuallySampler(propositions)]



# The LTLSampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getLTLSampler(sampler_id, propositions):

    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    if (tokens[0] == "Dataset"):
        dataset_name = tokens[1]
        shuffle = False if "no-shuffle" in tokens[2:] else True
        ids = None
        return DatasetSampler(propositions, dataset_name, shuffle)

    elif (tokens[0] == "SingleFormula"):
        return SingleFormulaSampler(propositions, None)

    elif (tokens[0] == "OrSampler"):
        return OrSampler(propositions)

    elif ("_OR_" in sampler_id): # e.g., Sequence_2_4_OR_UntilTask_3_3_1_1
        sampler_ids = sampler_id.split("_OR_")
        return OrSampler(propositions, sampler_ids)

    elif (tokens[0] == "Sequence"):
        return SequenceSampler(propositions, tokens[1], tokens[2])

    elif (tokens[0] == "Until"):
        return UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])

    elif (tokens[0] == "SuperSampler"):
        return SuperSampler(propositions)

    elif (tokens[0] == "Adversarial"):
        return AdversarialEnvSampler(propositions)

    elif (tokens[0] == "Eventually"):
        return EventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])

    elif (tokens[0] == "GlobalAvoidance"):
        return GlobalAvoidanceSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6])

    else: # "Default"
        return DefaultSampler(propositions)



REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(REPO_DIR, "datasets")

# sampler that reads the samplers from a precomputed dataset
# (computing the automata is too slow to be done online)
class DatasetSampler(LTLSampler):

    def __init__(self, propositions, dataset_name, shuffle=True, ids=None):

        dataset_folder = os.path.join(DATASETS_DIR, dataset_name)

        self.shuffle = shuffle
        self.ids = ids
        self.sampled_tasks = 0
        self.propositions = propositions

        self.current_id = None
        self.current_formula = None
        self.current_automaton = None

        # load config
        with open(os.path.join(dataset_folder, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)
        assert self.config["propositions"] == self.propositions
        self.n_prop = len(self.propositions)

        # load formulas
        with open(os.path.join(dataset_folder, 'formulas.pkl'), 'rb') as f:
            formulas = pickle.load(f)
        assert len(formulas) == self.config["n_formulas"]

        automata = [None] * self.config["n_formulas"]

        automata_path = os.path.join(dataset_folder, 'automata.pkl')
        self.has_automata = os.path.exists(automata_path)

        if self.has_automata:

            # load automata
            with open(automata_path, 'rb') as f:
                automata = pickle.load(f)
            assert len(automata) == self.config["n_formulas"]

            for i, automaton in enumerate(automata):

                transitions = automaton.transitions
                rewards = automaton.rewards

                rewards_vector = np.array(rewards, dtype=np.int64)
                states = sorted(transitions.keys())
                actions = sorted(next(iter(transitions.values())).keys())

                transitions_matrix = np.zeros((len(states), len(actions)+1), dtype=np.int64)
                for s, state in enumerate(states):
                    for a, action in enumerate(actions):
                        transitions_matrix[s,a] = transitions[state][action]
                    transitions_matrix[s,-1] = s  # self-loops

                automata[i] = {'transitions': transitions_matrix, 'rewards': rewards_vector}

        self.items = [{"formula": f, "automaton": a} for f, a in zip(formulas, automata)]

        # filter for ids
        if self.ids is not None:
            self.items = [self.items[i] for i in self.ids]

        self.n_tasks = len(self.items)
        self.order = np.arange(self.n_tasks)


    def sample(self):

        # shuffle at each cycle
        if self.shuffle and self.sampled_tasks % self.n_tasks == 0:
            np.random.shuffle(self.order)

        self.current_id = self.order[self.sampled_tasks % self.n_tasks]
        current_item = self.items[self.current_id]
        self.current_formula = current_item["formula"]
        self.current_automaton = current_item["automaton"]
        self.sampled_tasks += 1

        return self.current_formula


    def get_current_id(self):
        return self.current_id


    def get_current_formula(self):
        return self.current_formula


    def get_current_automaton(self):
        return self.current_automaton


    def get_formula(self, index):
        return self.items[index]['formula']


    def get_automaton(self, index):
        return self.items[index]['automaton']



class SingleFormulaSampler(LTLSampler):

    def __init__(self, propositions, formula):
        self.propositions = propositions
        self.formula = None


    def sample(self):
        return self.formula


    def set_formula(self, formula):
        self.formula = formula