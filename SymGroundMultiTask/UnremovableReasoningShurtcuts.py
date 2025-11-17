import sys
from itertools import product
import datetime



def check_alpha(dataset, alpha, phi, criterion):
    for trace in dataset:
        trace = list(trace)

        trace_alpha = substitute_map(trace,alpha, phi.alphabet)
        if criterion == "acceptance":
            #result_t = phi.accepts(trace)
            result_t = phi.accepts(substitute_map(trace, list(range(len(phi.alphabet))), phi.alphabet))
            result_t_a = phi.accepts(trace_alpha)
        elif criterion == "reward":
            #_, result_t = phi.process_trace(trace)
            _, result_t = phi.process_trace(substitute_map(trace, list(range(len(phi.alphabet))), phi.alphabet))
            _, result_t_a = phi.process_trace(trace_alpha)
        else:
            sys.exception("unrecognized criterion for counting RS: {}".format(criterion) )
        if result_t != result_t_a:
            return False
    return True

def check_alpha_easy_urss(alpha, phi):
    #print(phi.alphabet)
    #print(alpha)
    for q in range(phi.num_of_states):
        for i in range(len(phi.alphabet)):
            p = phi.alphabet[i]
            alpha_p = phi.alphabet[alpha[i]]
            if p != alpha_p:
                if phi.transitions[q][p] != phi.transitions[q][alpha_p]:
                    return False
    return True

def substitute_map(trace, alpha, alphabet):
    return list(map(lambda item: alphabet[alpha[item]], trace))


def find_reasoning_shortcuts(phi, mode="full"):
    start_time = datetime.datetime.now()

    reasoning_shortcuts = set()
    easy_urss = set()

    #one_step_traces = [[p] for p in phi.alphabet]
    one_step_traces = [[p] for p in range(len(phi.alphabet))]
    if mode == "full":
        alphas = set(product(list(range(len(phi.alphabet))), repeat= len(phi.alphabet)))
    elif mode == "no_renaming":
        alphas = set()
        for t in product(list(range(len(phi.alphabet))), repeat=len(phi.alphabet)):
            t_transf = tuple(substitute_map(list(range(len(phi.alphabet))), t, phi.alphabet))
            if len(set(t_transf)) < len(list(set(phi.alphabet))):
                alphas.add(t)

    else:
        sys.exit("Invalid mode for finding URS: ", mode)

    alphas = alphas.difference(easy_urss)

    D = {alpha: one_step_traces.copy() for alpha in alphas}

    iter = 0
    max_iter = 12


    #print("alphabet: ", phi.alphabet)
    #print("maps: ", alphas)
    #print("#maps: ", len(alphas))

    while D:# and iter < max_iter:
        #print("-----")
        #print(D)
        #print(iter)
        iter += 1
        next_D = {}
        for alpha in list(D.keys()):
            D_next_alpha = []
            #print("alpha: ", alpha)

            if check_alpha_easy_urss(alpha, phi):
                del D[alpha]
                easy_urss.add(alpha)

            else:
                if check_alpha(D[alpha], alpha, phi, "acceptance"):
                    # Expand dataset for the next iteration
                    # Check terminal states
                    for t in D[alpha]:
                        t_a = substitute_map(t, alpha, phi.alphabet)
                        #t_state, t_rew = phi.process_trace(t)
                        t_state, t_rew = phi.process_trace(substitute_map(t, list(range(len(phi.alphabet))), phi.alphabet))
                        t_a_state, t_a_rew = phi.process_trace(t_a)
                        t_state_terminal = (t_state in phi.absorbing_states)
                        t_a_state_terminal = (t_a_state in phi.absorbing_states)
                        if not t_state_terminal or not t_a_state_terminal:
                            # Check dummy transitions
                            #for p in phi.alphabet:
                            for p in range(len(phi.alphabet)):
                                t_prime = t + [p]
                                #t_pr_state, _ = phi.process_trace(t_prime)
                                t_pr_state, _ = phi.process_trace(substitute_map(t_prime, list(range(len(phi.alphabet))), phi.alphabet))
                                t_pr_a = substitute_map(t_prime, alpha, phi.alphabet)
                                t_pr_a_state, _ = phi.process_trace(t_pr_a)
                                if t_state != t_pr_state or t_a_state != t_pr_a_state:
                                    D_next_alpha.append(t_prime)
                else:
                    del D[alpha]

            if D_next_alpha:
                next_D[alpha] = D_next_alpha

        reasoning_shortcuts = set(D.keys()).union(easy_urss)
        D = next_D

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()

    #non restituisco le URS se ho terminato solo perchè arrivata alla max iter
    #if D:
    #    reasoning_shortcuts = set()

    return reasoning_shortcuts, execution_time

from ltlf2dfa.parser.ltlf import LTLfParser
def find_reasoning_shortcuts_naif(phi ):
    alphabet = phi.alphabet
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

