import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import transacc2pythomata


sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)



class ProbabilisticAutoma(nn.Module):

    metadata = {
        "initializations": ["gaussian"]
    }

    def __init__(self, num_actions, num_states, num_rewards, initialization="gaussian", device=None):
        super(ProbabilisticAutoma, self).__init__()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.num_actions = num_actions
        self.alphabet = [str(i) for i in range(num_actions)]
        self.num_states = num_states
        self.num_rewards = num_rewards
        self.reward_values = torch.tensor(list(range(num_rewards)))
        self.activation = sftmx_with_temp

        assert initialization in self.metadata["initializations"]

        if initialization == "gaussian":
            self.trans_prob = torch.empty(
                batch_size, num_actions, num_states, num_states,
                device=self.device, dtype=torch.float32
            ).normal_(mean=0, std=0.1)
            self.rew_matrix = torch.empty(
                batch_size, num_states, self.num_rewards,
                device=self.device, dtype=torch.float32
            ).normal_(mean=0, std=0.1)


    # input: sequence of actions (batch, length_seq, num_of_actions)
    def forward(self, action_seq, current_state= None):
        batch_size = action_seq.size()[0]
        length_size = action_seq.size()[1]

        pred_states = torch.zeros((batch_size, length_size, self.num_states))
        pred_rew = torch.zeros((batch_size, length_size, self.num_rewards))

        if current_state == None:
            s = torch.zeros((batch_size,self.num_states), device=device)
            s[:,0] = 1.0
        else:
            s = current_state

        for i in range(length_size):
            a = action_seq[:,i, :]
            s, r = self.step(s, a)
            pred_states[:,i,:] = s
            pred_rew[:,i,:] = r

        return pred_states, pred_rew


    def step(self, state, action):

        if type(action) == int:
            action= torch.tensor([action], dtype=torch.int32)

        trans_prob = self.trans_prob
        rew_matrix = self.rew_matrix

        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)

        selected_prob = torch.matmul(state.float(), trans_prob)
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
            action = torch.tensor([action], dtype=torch.int32)

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


    def net2dfa(self, min_temp):

        trans_prob = self.activation(self.trans_prob, min_temp)
        rew_matrix = self.activation(self.rew_matrix, min_temp)

        trans_prob = torch.argmax(trans_prob, dim= 2)
        rew_matrix = torch.argmax(rew_matrix, dim=1)

        trans = {}
        for s in range(self.num_states):
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
        pyautomaton = pyautomaton.reachable()
        pyautomaton = pyautomaton.minimize()
       
        return pyautomaton


    def initFromDfa(self, reduced_dfa, outputs, weight=10):
        with torch.no_grad():
            self.trans_prob[:,:,:] = 0
            self.rew_matrix[:,:] = 0

        # set the transition probabilities as the one in the dfa
        for s in reduced_dfa:
            for a in reduced_dfa[s]:
                with torch.no_grad():
                    self.trans_prob[a, s, reduced_dfa[s][a]] = 1

        # set reward matrix
        for s in range(len(reduced_dfa.keys())):
            with torch.no_grad():
                self.rew_matrix[s, outputs[s]] = weight



class MultiTaskProbabilisticAutoma(nn.Module):

    metadata = {
        "reward_types": ["boolean", "ternary"],
        "initializations": ["gaussian", "empty"]
    }

    def __init__(self, batch_size, num_actions, num_states, initialization="empty", reward_type="ternary", device=None):
        super(MultiTaskProbabilisticAutoma, self).__init__()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states

        assert reward_type in self.metadata["reward_types"]
        assert initialization in self.metadata["initializations"]

        if reward_type == "boolean":
            self.num_rewards = 2
            self.reward_values = torch.tensor([0, 1], device=self.device)
            self.reward_to_index = {0: 0, 1: 1}

        elif reward_type == "ternary":
            self.num_rewards = 3
            self.reward_values = torch.tensor([-1, 0, 1], device=self.device)
            self.reward_to_index = {-1: 0, 0: 1, 1: 2}

        if initialization == "gaussian":
            self.trans_prob = torch.empty(
                batch_size, num_actions, num_states, num_states,
                device=self.device, dtype=torch.float32
            ).normal_(mean=0, std=0.1)
            self.rew_matrix = torch.empty(
                batch_size, num_states, self.num_rewards,
                device=self.device, dtype=torch.float32
            ).normal_(mean=0, std=0.1)

        elif initialization == "empty":
            self.trans_prob = torch.empty(
                batch_size, num_actions, num_states, num_states,
                device=self.device, dtype=torch.float32
            )
            self.rew_matrix = torch.empty(
                batch_size, num_states, self.num_rewards,
                device=self.device, dtype=torch.float32
            )


    def forward(self, action_seq, current_state=None):

        batch_size, length_size, _ = action_seq.shape

        pred_states = torch.zeros((batch_size, length_size, self.num_states), device=self.device, dtype=torch.float32)
        pred_rew = torch.zeros((batch_size, length_size, self.num_rewards), device=self.device, dtype=torch.float32)

        if current_state is None:
            s = torch.zeros((batch_size, self.num_states), device=self.device, dtype=torch.float32)
            s[:, 0] = 1.0
        else:
            s = current_state.to(device=self.device, dtype=torch.float32)

        for i in range(length_size):
            a = action_seq[:, i, :]
            s, r = self.step(s, a)
            pred_states.select(1, i).copy_(s)
            pred_rew.select(1, i).copy_(r)

        return pred_states, pred_rew


    def step(self, state, action):

        # state: (batch, num_states)
        # action: (batch, num_actions)
        # trans_prob: (batch, num_actions, num_states, num_states)
        # rew_matrix: (batch, num_states, num_rewards)

        selected_prob = torch.einsum('bs,basn->ban', state, self.trans_prob)  # (batch, num_actions, num_states)
        next_state = torch.einsum('ba,bas->bs', action, selected_prob)  # (batch, num_states)
        next_reward = torch.einsum('bs,bsr->br', next_state, self.rew_matrix)  # (batch, num_rewards)

        return next_state, next_reward


    def initFromDfas(self, dfas_transitions, dfas_rewards, weight=10):

        with torch.no_grad():

            self.trans_prob.zero_()
            self.rew_matrix.zero_()

            for index, (trans, rews) in enumerate(zip(dfas_transitions, dfas_rewards)):

                num_states = trans.shape[0]
                trans = trans.transpose(0,1)

                rews_idx = torch.empty_like(rews, dtype=torch.int64, device=self.device)
                for val, idx in self.reward_to_index.items():
                    rews_idx[rews == val] = idx

                self.trans_prob[index, :, :num_states, :num_states].scatter_(2, trans.unsqueeze(-1), 1.0)
                self.rew_matrix[index, :num_states].scatter_(1, rews_idx.unsqueeze(-1), weight)