import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import dot2pythomata, transacc2pythomata

from .Minimization import MinimizableMooreMachine

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)

class ProbabilisticAutoma(nn.Module):
    def __init__(self, numb_of_actions, numb_of_states, numb_of_rewards, initialization="gaussian"):
        super(ProbabilisticAutoma, self).__init__()
        self.numb_of_actions = numb_of_actions
        self.alphabet = [str(i) for i in range(numb_of_actions)]
        self.numb_of_states = numb_of_states
        self.numb_of_rewards = numb_of_rewards
        self.reward_values = torch.Tensor(list(range(numb_of_rewards)))
        self.activation = sftmx_with_temp
        #if initialization == "gaussian":
        #standard gaussian noise initialization
        # --- FIX 1: USE nn.Parameter ---
        # We must wrap the tensors so the Optimizer sees them!
        self.trans_prob = nn.Parameter(
            torch.normal(0, 5.0, size=(numb_of_actions, numb_of_states, numb_of_states))
        )
        self.rew_matrix = nn.Parameter(
            torch.normal(0, 5.0, size=(numb_of_states, numb_of_rewards))
        )
        
        '''
        if initialization == "random_DFA":
            random_dfa = Random_DFA(self.numb_of_states, self.numb_of_actions)
            transitions = random_dfa.transitions
            final_states = []
            for s in range(self.numb_of_states):
                if random_dfa.acceptance[s]:
                    final_states.append(s)
            self.initFromDfa(transitions, final_states)
        '''

    #input: sequence of actions (batch, length_seq, num_of_actions)
    def forward(self, action_seq, temp, current_state=None):
        batch_size = action_seq.size()[0]
        length_size = action_seq.size()[1]
        
        # FIX: Get the device from the input so tensors are created on GPU/CPU dynamically
        device = action_seq.device 

        # FIX: Add device=device argument
        pred_states = torch.zeros((batch_size, length_size, self.numb_of_states), device=device)
        pred_rew = torch.zeros((batch_size, length_size, self.numb_of_rewards), device=device)

        if current_state is None:
            # FIX: Use the captured device variable
            s = torch.zeros((batch_size, self.numb_of_states), device=device)
            # initial state is 0 for construction
            s[:, 0] = 1.0
        else:
            s = current_state

        for i in range(length_size):
            a = action_seq[:, i, :]
            s, r = self.step(s, a, temp)
            
            pred_states[:, i, :] = s
            pred_rew[:, i, :] = r
            
        return pred_states, pred_rew

    def step(self, state, action, temp):
        
        # --- FIX 2: Decouple Matrices Temp ---
        # Even if the Symbol Grounder is exploring (High Temp), 
        # the machine logic should be relatively sharp (Low Temp).
        # We clamp the matrix temp to max 1.0.
        matrix_temp = min(temp, 1.0)
        
        
        if type(action) == int:
            action= torch.IntTensor([action])
        #activation
        
        # 1. Get Probability Matrices
        # Shape: (Actions, Current_State, Next_State)
        T = self.activation(self.trans_prob, matrix_temp)
        # Shape: (Current_State, Rewards)
        R = self.activation(self.rew_matrix, matrix_temp)
        
        # 2. Calculate Transition (The "Logic" Step)
        # We use Einstein Summation for clarity and safety.
        # b=batch, a=action, s=current_state, k=next_state
        
        # Logic: 
        # We want to find the distribution of the NEXT state (k).
        # We take the current state (s) AND the action taken (a).
        # We look up the transition tensor T[a, s, k].
        
        # Formula: Sum over 'a' and 's'
        next_state = torch.einsum('bs, ba, ask -> bk', state, action, T)
        
        # 3. Calculate Reward
        # Logic: Based on the state we just landed in (next_state), what is the reward?
        next_reward = torch.matmul(next_state, R)
        
        return next_state, next_reward

    def step_(self, state, action, temp):

        print("##############################")
        print("state: ", state)
        print("state size: ", state.size())
        print("action :", action)
        print("action size :", action.size())

        print("trans prob size:", self.trans_prob.size())
        print("trans prob:", self.trans_prob)

        if type(action) == int:
            action = torch.IntTensor([action])


        #no activation
        trans_prob = self.trans_prob
        rew_matrix = self.rew_matrix

        print("trans_prob activated size: ", trans_prob.size())
        print("trans_prob activated: ", trans_prob)
        print("rew matrix size:", self.rew_matrix.size())
        print("rew matrix:", self.rew_matrix)
        print("rew_matrix activated size: ", rew_matrix.size())
        print("rew_matrix activated: ", rew_matrix)

        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)

        print("transprob size: ", trans_prob.size())
        print("state size: ", state.size())

        selected_prob = torch.matmul(state, trans_prob)

        print("selected prob size: ", selected_prob.size())
        print("selected prob: ", selected_prob)

        next_state = torch.matmul(action.unsqueeze(1), selected_prob.squeeze())

        print("next_state size:", next_state.size())
        print("next_state :", next_state)
        print("rew_matrix:", rew_matrix)

        next_reward = torch.matmul(next_state, rew_matrix)

        print("next reward:", next_reward)
        print("next_rew size: ", next_reward.size())


        return next_state.squeeze(1), next_reward.squeeze(1)

    def net2dfa(self, min_temp, name_automata = None):

        trans_prob = self.activation(self.trans_prob, min_temp)
        rew_matrix = self.activation(self.rew_matrix, min_temp)

        last_label = rew_matrix[-1]
        print("last label: ", last_label)
        print(rew_matrix.size())

        trans_prob = torch.argmax(trans_prob, dim= 2)
        
        rew_matrix = torch.argmax(rew_matrix, dim=1)
        
        print(rew_matrix.size())
       
       #TODO
        

        #2transacc
        trans = {}
        for s in range(self.numb_of_states):
            trans[s] = {}
        acc = []

    


        for i, rew in enumerate(rew_matrix):
                if rew == 2:        #da controllare e chiedere ad elena
                    acc.append(True)
                else:
                    acc.append(False)
        for a in range(trans_prob.size()[0]):
            for s, s_prime in enumerate(trans_prob[a]):
                    trans[s][str(a)] = s_prime.item()

     
        pyautomaton = transacc2pythomata(trans, acc, self.alphabet)
        print(f"Saving automata in {name_automata}.dot")
        if name_automata is not None:
            pyautomaton.to_graphviz().render(name_automata + ".dot")
      
 
        pyautomaton = pyautomaton.reachable()
        

        pyautomaton = pyautomaton.minimize()

        pyautomaton, deleted_symbols, alphabet = MinimizableMooreMachine(pyautomaton).return_minimized_pydfa()


        print("Deleted symbols: ", deleted_symbols)



        if name_automata is not None:
            pyautomaton.to_graphviz().render(name_automata + "_minimized.dot")
        
       
        #self.dfa.to_graphviz().render(self.automata_dir + self.formula_name + "_exp" + str(self.exp_num) + "_minimized_"+mode+".dot")
        

        #TODO: 30 stati e poi aumentare i simboli       2030
        #salvare il DFA subito dopo il DFA


        return pyautomaton


    def initFromDfa(self, reduced_dfa, outputs, weigth=10):
        with torch.no_grad():
            #zeroing transition probabilities
            for a in range(self.numb_of_actions):
                for s1 in range(self.numb_of_states):
                    for s2 in range(self.numb_of_states):
                        self.trans_prob[a, s1, s2] = 0.0

            #zeroing reward matrix
            for s in range(self.numb_of_states):
                for r in range(self.numb_of_rewards):
                    self.rew_matrix[s,r] = 0.0


        #set the transition probabilities as the one in the dfa
        for s in reduced_dfa:
            for a in reduced_dfa[s]:
                with torch.no_grad():
                    self.trans_prob[a, s, reduced_dfa[s][a]] = weigth

        #set reward matrix
        for s in range(len(reduced_dfa.keys())):
                with torch.no_grad():
                    self.rew_matrix[s, outputs[s]] = weigth
