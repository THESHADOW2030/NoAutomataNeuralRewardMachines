import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import dot2pythomata, transacc2pythomata

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
        self.trans_prob = torch.normal(0, 0.5, size=( numb_of_actions, numb_of_states, numb_of_states), requires_grad=True, device=device)
        self.trans_prob = self.trans_prob.double()
        self.rew_matrix = torch.normal(0, 0.5, size=( numb_of_states, numb_of_rewards), requires_grad=True, device=device)
        self.rew_matrix = self.rew_matrix.double()
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
    def forward(self, action_seq, temp, current_state= None):
        batch_size = action_seq.size()[0]
        length_size = action_seq.size()[1]

        pred_states = torch.zeros((batch_size, length_size, self.numb_of_states))
        pred_rew = torch.zeros((batch_size, length_size, self.numb_of_rewards))

        if current_state == None:
            s = torch.zeros((batch_size,self.numb_of_states)).to(device)
            #initial state is 0 for construction
            s[:,0] = 1.0
        else:
            s = current_state
        #print("current state: ", s[0])
        for i in range(length_size):
            a = action_seq[:,i, :]
            #print("current symbol: ", a[0])

            s, r = self.step(s, a, temp)
            #print("Reward: ", r[0])
            #print("current state: ", s[0])
            s = sftmx(s)
            pred_states[:,i,:] = s
            pred_rew[:,i,:] = r
        #assert False
        return pred_states, pred_rew

    def step(self, state, action, temp):
        
        if type(action) == int:
            action= torch.IntTensor([action])
        #activation
        #trans_prob = self.activation(self.trans_prob, temp)
        #rew_matrix = self.activation(self.rew_matrix, temp)
        #no activation
        trans_prob = self.trans_prob
        rew_matrix = self.rew_matrix
      
        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)

        selected_prob = torch.matmul(state.double(), trans_prob)

        next_state = torch.matmul(action.unsqueeze(1), selected_prob.squeeze())
      
        next_reward = torch.matmul(next_state, rew_matrix)
       
        return next_state.squeeze(1), next_reward.squeeze(1)

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
                if rew == 0:
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
