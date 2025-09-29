import sys
from itertools import product
import datetime

def check_alpha(dataset, alpha, phi, criterion):
    for trace in dataset:
        trace = list(trace)
        trace_alpha = substitute_map(trace,alpha)
        if criterion == "acceptance":
            result_t = phi.accepts(trace)
            result_t_a = phi.accepts(trace_alpha)
        elif criterion == "reward":
            _, result_t = phi.process_trace(trace)
            _, result_t_a = phi.process_trace(trace_alpha)
        else:
            sys.exception("unrecognized criterion for counting RS: {}".format(criterion) )
        if result_t != result_t_a:
            return False
    return True


def substitute_map(trace, alpha):
    return list(map(lambda item: alpha[item], trace))



def find_reasoning_shortcuts(phi):
    start_time = datetime.datetime.now()

    if -100 in phi.rewards:
        terminal_rew = -100
    else:
        terminal_rew = 100
    print("TERMINAL REWARD:", terminal_rew)
    one_step_traces = [[p] for p in phi.alphabet]
    alphas = set(product(phi.alphabet, repeat= len(phi.alphabet)))
    D = {alpha: one_step_traces.copy() for alpha in alphas}

    while D:
        next_D = {}
        for alpha in list(D.keys()):
            D_next_alpha = []

            if check_alpha(D[alpha], alpha, phi, "acceptance"):
                # Expand dataset for the next iteration
                # Check terminal states
                for t in D[alpha]:
                    t_a = substitute_map(t, alpha)
                    t_state, t_rew = phi.process_trace(t)
                    t_a_state, t_a_rew = phi.process_trace(t_a)
                    t_state_terminal = (t_rew == terminal_rew)
                    t_a_state_terminal = (t_a_rew == terminal_rew)
                    if not t_state_terminal or not t_a_state_terminal:
                        # Check dummy transitions
                        for p in phi.alphabet:
                            t_prime = t + [p]
                            t_pr_state, _ = phi.process_trace(t_prime)
                            t_pr_a = substitute_map(t_prime, alpha)
                            t_pr_a_state, _ = phi.process_trace(t_pr_a)
                            if t_state != t_pr_state or t_a_state != t_pr_a_state:
                                D_next_alpha.append(t_prime)
            else:
                del D[alpha]

            if D_next_alpha:
                next_D[alpha] = D_next_alpha

        reasoning_shortcuts = set(D.keys())
        D = next_D

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()

    return reasoning_shortcuts, execution_time

from ltlf2dfa.parser.ltlf import LTLfParser
def find_reasoning_shortcuts_naif(phi, alphabet ):
    start_time = datetime.datetime.now()
    #put the declare condition
    #phi = formula string
    # alphabet = list of characters
    rs = set()

    alphas = set(product(alphabet, repeat= len(alphabet)))
    count = 0
    for alpha in alphas:
        #print("map:", alpha)
        phi_alpha = substitute_map_string(phi, alpha)
        #print("new formula:", phi_alpha)

        equivalence = "(({})->({})) & (({})->({}))".format(phi, phi_alpha, phi_alpha, phi)

        print(equivalence)
        parser = LTLfParser()
        formula_str = equivalence
        formula = parser(formula_str)
        dfa = formula.to_dfa()
        print(dfa)
        if check_equivalence(dfa):
            rs.add(alpha)

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()

    return rs, execution_time

def check_equivalence(dfa_string):
    return dfa_string == 'digraph MONA_DFA {\n rankdir = LR;\n center = true;\n size = "7.5,10.5";\n edge [fontname = Courier];\n node [height = .5, width = .5];\n node [shape = doublecircle]; 1;\n node [shape = circle]; 1;\n init [shape = plaintext, label = ""];\n init -> 1;\n 1 -> 1 [label="true"];\n}'

def substitute_map_string(trace, alpha):
    #trace = str(trace)

    #print(trace)
    #print(alpha)
    l= list(map(lambda item: sub_char(item, alpha), trace))
    new_string = ""
    for char in l:
        new_string += char
    return new_string
    #print("new trace: ", trace)
    #for i, rep in enumerate(alpha):
    #    trace = trace.replace(i, rep)
    #    print(trace)
def sub_char(item, alpha):
    try:
        return str(alpha[int(item)])
    except:
        return item

