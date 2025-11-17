from graphviz import Source
from pythomata import SimpleDFA
import numpy as np
from ltlf2dfa.parser.ltlf import LTLfParser

from SymGroundMultiTask.utils import dot2pythomata, shift_back_nodes


class DFA:

    # 3 types of initialization:
    # init_from_ltl: arg1 -> ltl_formula | arg2 -> num_symbols | arg3 -> formula_name
    # random_init: arg1 -> num_states | arg2 -> num_symbols
    # init_from_transacc: arg1 -> transitions | arg2 -> acceptances
    def __init__(self, arg1, arg2, arg3, dictionary_symbols):

        self.dictionary_symbols = dictionary_symbols

        if isinstance(arg1, str):
            self.init_from_ltl(arg1, arg2, arg3, dictionary_symbols)
        elif isinstance(arg1, int):
            self.random_init(arg1, arg2)
        elif isinstance(arg1, dict):
            self.init_from_transacc(arg1, arg2)
        else:
            raise Exception("Uncorrect type for the argument initializing th DFA: {}".format(type(arg1)))

        self.calculate_absorbing_states()
        self.calculate_live_states()


    def calculate_absorbing_states(self):
        self.absorbing_states = []
        for q, transitions in self.transitions.items():
            if all(dest == q for dest in transitions.values()):
                self.absorbing_states.append(q)



    def calculate_live_states(self):

        self.liveliness = [self.acceptance[q] for q in range(self.num_of_states)]

        changed = True
        while changed:
            changed = False
            for q in range(self.num_of_states):
                if not self.liveliness[q]:
                    for s in self.transitions[q]:
                        next_q = self.transitions[q][s]
                        if self.liveliness[next_q]:
                            self.liveliness[q] = True
                            changed = True
                            break


    def init_from_ltl(self, ltl_formula, num_symbols, formula_name, dictionary_symbols, save=False):

        # convert LTL formula into DFA (dot string)
        parser = LTLfParser()
        ltl_formula_parsed = parser(ltl_formula)
        dot_dfa = ltl_formula_parsed.to_dfa()
        dot_dfa = shift_back_nodes(dot_dfa)

        # save symbolic DFA
        if save:
            with open(f"symbolicDFAs/{formula_name}_symbolic.dot", "w") as f:
                f.write(dot_dfa)
            s = Source.from_file(f"symbolicDFAs/{formula_name}_symbolic.dot")
            s.render(f"symbolicDFAs/{formula_name}_symbolic", format='pdf', cleanup=True, view=True)

        # convert dot file into SymbolicDFA
        dfa = dot2pythomata(dot_dfa, dictionary_symbols)
        # print(dfa.__dict__)

        # from symbolic DFA to simple DFA
        self.alphabet = dictionary_symbols
        self.transitions = self.reduce_dfa(dfa)

        # convert final states
        self.num_of_states = len(self.transitions)
        final_states = set(dfa._final_states)
        self.acceptance = [s in final_states for s in range(self.num_of_states)]

        # complete the transitions with the symbols that ARE NOT in the formula
        self.num_of_symbols = len(dictionary_symbols)
        self.alphabet = list(range(self.num_of_symbols))
        if len(self.transitions[0]) < self.num_of_symbols:
            for s in range(self.num_of_states):
                for sym in self.alphabet:
                    if sym not in self.transitions[s].keys():
                        self.transitions[s][sym] = s

        # save final DFA
        if save:
            self.write_dot_file(f"symbolicDFAs/{formula_name}.dot")
            s = Source.from_file(f"symbolicDFAs/{formula_name}.dot")
            s.render(f"symbolicDFAs/{formula_name}", format='pdf', cleanup=True, view=True)


    def reduce_dfa(self, dfa):

        admissible_transitions = [
            {sym: (sym == a) for sym in self.alphabet}
            for a in self.alphabet
        ]

        reduced = {}

        for state in dfa._states:
            reduced[state] = {}
            transitions = dfa._transition_function[state]
            for target_state, symbolic_condition in transitions.items():
                for sym_idx, substitution in enumerate(admissible_transitions):
                    if symbolic_condition.subs(substitution):
                        reduced[state][sym_idx] = target_state

        return reduced


    def init_from_transacc(self, trans, acc):
        self.num_of_states = len(acc)
        self.num_of_symbols = len(trans[0])
        self.transitions = trans
        self.acceptance = acc
        self.alphabet = list(range(self.num_of_symbols))


    def random_init(self, num_of_states, num_of_symbols):
        self.num_of_states = num_of_states
        self.num_of_symbols = num_of_symbols

        transitions = {}
        acceptance = np.random.randint(0, 2, size=num_of_states, dtype=bool).tolist()

        for s in range(num_of_states):
            trans_from_s = {}

            if s < num_of_states - 1:
                s_prime = np.random.randint(s + 1, num_of_states)
                a_start = np.random.randint(num_of_symbols)
                trans_from_s[a_start] = s_prime
            else:
                a_start = None

            # Fill in the rest
            for a in range(num_of_symbols):
                if a != a_start:
                    trans_from_s[a] = np.random.randint(num_of_states)

            transitions[s] = trans_from_s.copy()

        self.transitions = transitions
        self.acceptance = acceptance
        self.alphabet = list(range(num_of_symbols))


    def accepts(self, string):
        return self.accepts_from_state(0, string)


    def accepts_from_state(self, state, string):
        for a in string:
            state = self.transitions[state][a]
        return self.acceptance[state]


    def to_pythomata(self):
        trans = self.transitions
        acc = self.acceptance
        accepting_states = set()
        for i in range(len(acc)):
            if acc[i]:
                accepting_states.add(i)

        automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

        return automaton


    def to_dot_str(self):
        dot_str = (
            "digraph MONA_DFA {\n"
            "rankdir = LR;\n"
            "center = true;\n"
            "size = \"7.5,10.5\";\n"
            "edge [fontname = Courier];\n"
            "node [height = .5, width = .5];\n"
            "node [shape = doublecircle];"
        )
        for i, rew in enumerate(self.acceptance):
            if rew:
                dot_str += str(i) + ";"
        dot_str += (
            "\nnode [shape = circle]; 0;\n"
            "init [shape = plaintext, label = \"\"];\n"
            "init -> 0;\n"
        )

        for s in range(self.num_of_states):
            for a in range(self.num_of_symbols):
                s_prime = self.transitions[s][a]
                dot_str += "{} -> {} [label=\"{}\"];\n".format(s, s_prime, self.dictionary_symbols[a])

        dot_str += "}\n"
        return dot_str


    def write_dot_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(self.to_dot_str())


    def show(self, save_path=None):
        dot_dfa = self.to_dot_str()
        s = Source(dot_dfa)
        s.render(save_path, format='pdf', cleanup=True, view=True)



class MooreMachine(DFA):

    def __init__(self, arg1, arg2, arg3, dictionary_symbols, reward = "distance"):
        super().__init__(arg1, arg2, arg3, dictionary_symbols)

        self.rewards = [0 for _ in range(self.num_of_states)]

        # associate reward based on the distance from a "final state"
        if reward == "distance":

            # starts with 0 on final states, 100 otherwise
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 0
                else:
                    self.rewards[s] = 100

            # propagate with fixpoint algorithm
            old_rew = self.rewards.copy()
            termination = False
            while not termination:
                termination = True
                for s in range(self.num_of_states):
                    if not self.acceptance[s]:
                        next = [ self.rewards[self.transitions[s][sym]] for sym in self.alphabet if self.transitions[s][sym] != s]
                        if len(next) > 0:
                            self.rewards[s] = 1 + min(next)

                termination = (str(self.rewards) == str(old_rew))
                old_rew = self.rewards.copy()

            for i in range(len(self.rewards)):
                self.rewards[i] *= -1
            minimum = min([r for r in self.rewards if r != -100])

            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = (r - minimum)

            # rescale to have maximum to 100
            maximum = max(self.rewards)
            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = 100 * r / maximum

        # binary reward: 1 for "final state", 0 otherwise
        elif reward == "acceptance":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 1
                else:
                    self.rewards[s] = 0

        # ternary reward: 1 for "final state", -1 for dead states, 0 otherwise
        elif reward == "ternary":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 1
                elif not self.liveliness[s]:
                    self.rewards[s] = -1
                else:
                    self.rewards[s] = 0

        else:
            raise Exception("Reward based on '{}' NOT IMPLEMENTED".format(reward))


    # returns the last reward produced by the MooreMachine for the input string
    def process_trace(self, string, state=0):
        a = string[0]
        next_state = self.transitions[state][a]

        if len(string) == 1:
            return next_state, self.rewards[next_state]

        return self.process_trace(string[1:], next_state)