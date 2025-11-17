import torch
import random
import os
import io
import re
import numpy as np
from numpy.random import RandomState
from pythomata import SymbolicAutomaton, SimpleDFA


def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


# shift the nodes names back by 1
def shift_back_nodes(dot_dfa):
    def shift(match):
        return str(int(match.group(0)) - 1)
    dot_dfa = re.sub(r'\b\d+\b', shift, dot_dfa)
    return dot_dfa


def dot2pythomata(dot_str, action_alphabet):

        # read dot string
        dot_file = io.StringIO(dot_str)
        Lines = dot_file.readlines()

        states = set()

        # find all states
        count = 0
        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                states.add(line.strip().split(" ")[0])
            # capture the final states
            elif "doublecircle" in line.strip():
                final_states = line.strip().split(';')[1:-1]
                final_states = [s.strip() for s in final_states]

        # keep same names
        states = list(states)
        states.sort()

        automaton = SymbolicAutomaton()

        # create all states
        state_dict = dict()
        state_dict['0'] = 0
        for state in states:
            if state == '0':
                continue
            state_dict[state] = automaton.create_state()

        # set initial state (always 0)
        automaton.set_initial_state(state_dict['0'])
        # set final states
        for state in final_states:
            automaton.set_accepting_state(state_dict[state], True)

        # add all transitions
        count = 0
        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                init_state = state_dict[line.strip().split(" ")[0]]
                action = line.strip().split('"')[1]
                final_state = state_dict[line.strip().split(" ")[2]]
                automaton.add_transition((init_state, action, final_state))

        return automaton


def transacc2pythomata(trans, acc, action_alphabet):
    accepting_states = set()
    for i in range(len(acc)):
        if acc[i]:
            accepting_states.add(i)

    automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

    return automaton


def eval_acceptance(classifier, automa, alphabet, dataset, automa_implementation='dfa', temperature = 1.0, discretize_labels= False, mutually_exc_sym=True, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # automa implementation =
    #   - 'dfa' use the perfect dfa given
    #   - 'lstm' use the lstm model
    #   - 'logic_circuit' use the fuzzy automaton

    total = 0
    correct = 0
    test_loss = 0
    classifier.eval()
    numb_of_symbols = len(alphabet)
    with torch.no_grad():
        for i in range(len(dataset[0])):
            image_sequences = dataset[0][i].to(device)
            
            labels = dataset[1][i].to(device)
            batch_size = image_sequences.size()[0]

            length_seq = image_sequences.size()[1]

            num_channels = image_sequences.size()[2]

            if len(image_sequences.size()) > 3:
                pixels_v = image_sequences.size()[3]
                pixels_h = image_sequences.size()[4]
                symbols = classifier(image_sequences.view(-1, num_channels, pixels_v, pixels_h))
            else:
                symbols = classifier(image_sequences.view(-1, num_channels).float())
            '''
            if discretize_labels:
                symbols[:,0] = torch.where(symbols[:,0] > 0.5, 1., 0.)
                symbols = sftmx_with_temp(symbols, temp=0.00001)
            '''
            sym_sequences = symbols.view(batch_size, length_seq, numb_of_symbols)

            if automa_implementation == 'lstm':
                accepted = automa(sym_sequences)
                accepted = accepted[-1]

                output = torch.argmax(accepted).item()
            elif automa_implementation == 'logic_circuit':

                pred_states, pred_rew = automa(sym_sequences, temperature)
                num_out = pred_rew.size()[-1]
                pred_rew = pred_rew.view(-1, num_out)
                labels = labels.view(-1)
               
                output = torch.argmax(pred_rew, dim=-1).to(device)
              
            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)
        
            total += labels.size()[0]
          
            correct += (output==labels).sum().item()
        test_accuracy = 100. * correct/(float)(total)
    return test_accuracy


def eval_learnt_DFA_acceptance(automa, dataset, automa_implementation='logic_circuit', temp=1.0, alphabet=None, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # automa implementation =
    #   - 'dfa' use the discretized probabilistic automaton #TODO
    #   - 'logic_circuit'
    #   - 'lstm' use the lstm model in automa

    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i in range(len(dataset[0])):
            sym = dataset[0][i].to(device)
            if automa_implementation != "dfa":
                label = dataset[1][i].to(device)
            else:
                label = dataset[1][i]

            if automa_implementation == 'logic_circuit' or automa_implementation == 'lstm':
                pred_acceptace = automa(sym, temp)
                output = torch.argmax(pred_acceptace, dim= 1)
            elif automa_implementation == 'dfa':

                output = torch.zeros((sym.size()[0]), dtype=torch.int32)
                for k in range(sym.size()[0]):

                    sym_trace = tensor2string(sym[k])
                    output[k] = int(automa.accepts(sym_trace))

            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)
            total += output.size()[0]

            correct += sum(output==label).item()

            accuracy = 100. * correct/(float)(total)

    return accuracy


def eval_image_classification_from_traces(traces_images, traces_labels, classifier, mutually_exclusive, return_errors=False, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    total = 0
    correct = 0
    classifier.eval()
    errors = torch.zeros((0,2), device=device)

    LEN = min(len(traces_images),len(traces_labels))
 
    with (torch.no_grad()):
        for i in range(LEN) :
            batch_t_sym = traces_labels[i].to(device)
            batch_t_img = traces_images[i].to(device)
            batch_size= batch_t_img.size()[0]
            length_seq = batch_t_img.size()[1]
            num_channels = batch_t_img.size()[2]
            if len(batch_t_img.size()) > 3:
                pixels_v, pixels_h = list(batch_t_img.size())[3:]
                pred_symbols = classifier(batch_t_img.view(-1, num_channels, pixels_v, pixels_h))
            else:
                pred_symbols = classifier(batch_t_img.view(-1, num_channels).float())

            gt_symbols = batch_t_sym.view(-1, batch_t_sym.size()[-1])
            if  not mutually_exclusive:

                y1 = torch.ones(batch_t_sym.size(), device=device)
                y2 = torch.zeros(batch_t_sym.size(), device=device)

                output_sym = pred_symbols.where(pred_symbols <= 0.5, y1)
                output_sym = output_sym.where(pred_symbols > 0.5, y2)

                correct += torch.sum(output_sym == batch_t_sym).item()
                total += torch.numel(pred_symbols)

            else:
                output_sym = torch.argmax(pred_symbols, dim=1)
                gt_sym = torch.argmax(gt_symbols, dim = 1)
                equality = output_sym == gt_sym
                correct += torch.sum(equality).item()
                if return_errors:
                    eq_list = list(equality)
                    for eq_i,eq in enumerate(eq_list):
                        if not eq:
                            errors = torch.cat((errors, pred_symbols[eq_i,:].unsqueeze(0)), dim=0)
                total += torch.numel(output_sym)

    accuracy = 100. * correct / (float)(total)
    if return_errors:
        return accuracy, errors
    return accuracy


def pprint_ltl_formula(formula, indentation=0):
    if isinstance(formula, tuple):
        print('    '*indentation + "(" + formula[0])
        formula = formula[1:]
        for item in formula:
            pprint_ltl_formula(item, indentation+1)
        print('    '*indentation + ")")
    else:
        print('    '*indentation + formula)