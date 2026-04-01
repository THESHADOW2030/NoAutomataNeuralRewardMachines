import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm

from .NN_models import ActorCritic, RNN, Net
from .NRM.NeuralRewardMachine import NeuralRewardMachine
from .NRM.utils import eval_acceptance

from collections import deque
import cv2
import os
import pickle
import bisect
import itertools

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
torch.autograd.set_detect_anomaly(True)

class TraceReplayBuffer_old:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def add(self, traj, labels, loss = 0):
        self.data.append((traj, labels))

    def __len__(self):
        return len(self.data)

    def sample(self, n):
        if len(self.data) == 0:
            return []
        idxs = np.random.choice(
            len(self.data), size=min(n, len(self.data)), replace=False
        )
        return [self.data[i] for i in idxs]

class TraceReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.data = []
        self.counter = itertools.count() 

    def add(self, traj, labels, loss=0):
        count = next(self.counter)
        bisect.insort(self.data, (loss, count, traj, labels))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def sample(self, n):
        n = min(n, len(self.data))
        if n == 0:
            return []
        sampled = []
        for _ in range(n):
            loss, _, traj, labels = self.data.pop()
            sampled.append((traj, labels))
        return sampled

# --- HYPERPARAMS ---
max_episodes = 10000
rnn_outputs = 5
num_layers = 2
hidden_size = 120
rnn_hidden_size = 50
slide_wind = 100
lr = 0.0004

# --- PPO SPECIFIC HYPERPARAMS ---
ppo_epochs = 4            # Number of optimization epochs per batch
clip_param = 0.2          # PPO clip parameter
max_grad_norm = 0.5       # Gradient clipping

# we train the policy every num_steps
num_steps = 5
TT_policy = 1  
TT_grounder = 10 
grounder_epochs = 150
TTT = 10
target_batch_grounder = 1024

def plot_smoothed_sequence_classification_accuracy():
    for i in range(5):
        file_path = os.path.join(os.path.dirname(__file__), f"sequence_classification_accuracy_{i}.txt")
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()]
        lines = list(map(float, lines))
        smoothed = []
        mean_val = 0
        for idx, line in enumerate(lines):
            if idx % 100 == 0:
                smoothed.append((idx, mean_val / 100))
                mean_val = 0
            mean_val += line
            if idx == len(lines) - 1:
                smoothed.append((idx, mean_val / (idx % 100 + 1))) 
        lines = smoothed

        plt.plot(*zip(*lines))
        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Sequence Classification Accuracy")
        plt.title("Smoothed Sequence Classification Accuracy over Training Steps")
        plt.grid()
        plt.savefig(
            os.path.join(
                os.path.dirname(__file__),
                f"sequence_classification_accuracy_smoothed_{i}.png",
            )
        )
        plt.close()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        m = masks[step].to(device)
        A = rewards[step].to(device)
        B = gamma * R * m
        R = A + B
        returns.insert(0, R)
    return returns

def pad_list(lst, desired_length):
    if len(lst) < desired_length and lst:
        lst.extend([lst[-1]] * (desired_length - len(lst)))
    return lst

def prepare_dataset(sequence_accuracy, image_trajectory, info_trajectory, TT):
    indices = list(np.argsort(sequence_accuracy))
    indices.reverse()
    indices = indices[: int(TT / 2)]
    worst_trajectories = [image_trajectory[i] for i in indices]
    worst_related_info = [info_trajectory[i] for i in indices]
    return worst_trajectories, worst_related_info

def recurrent_A2C(
    env,
    path,
    experiment,
    method,
    feature_extraction,
    num_of_states=None,
    num_of_symbols=None,
    hidden_size_rnn=50,
    formula_name=""
):
    task_number = int(formula_name.split(" ")[0][-2])
    if task_number in [1, 3, 5, 6, 7, 8]:
        max_traces = 2
    elif task_number in [2, 4]:
        max_traces = 3
    else:
        raise ValueError("Invalid task number in formula name.")
    
    f = open(path + "/train_rewards_" + str(experiment) + ".txt", "w")
    f.close()

    rnn_hidden_size = hidden_size_rnn

    (
        num_of_states_override,
        num_of_symbols_override,
        num_automaton_outputs,
        transition_function,
        automaton_rewards,
    ) = env.get_automaton_specs()

    if num_of_states is None:
        num_of_states = num_of_states_override
    if num_of_symbols is None:
        num_of_symbols = num_of_symbols_override

    num_outputs = env.action_space.n

    params = []
    if feature_extraction:
        cnn = Net().to(device)
        cnn.float() 
        CNN_output_size = 16
        num_inputs = CNN_output_size
        params += list(cnn.parameters())
    else:
        num_inputs = env.state_space_size

    if method == "rnn":
        model = ActorCritic(rnn_hidden_size, num_outputs, hidden_size).to(device)
    else:
        model = ActorCritic(num_inputs + num_of_states, num_outputs, hidden_size).to(device)

    params += list(model.parameters())
    model.float() 

    if method == "rnn":
        rnn = RNN(num_inputs, rnn_hidden_size, num_layers).to(device)
        rnn.float() 
        params += list(rnn.parameters())
    elif method == "nrm":
        f = open(path + "/sequence_classification_accuracy_" + str(experiment) + ".txt", "w")
        f.close()
        f = open(path + "/image_classification_accuracy_" + str(experiment) + ".txt", "w")
        f.close()

        if env.state_type == "symbol":
            dataset = "minecraft_location"
        elif env.state_type == "image":
            dataset = "minecraft_image"

        grounder = NeuralRewardMachine(
            num_of_states,
            num_of_symbols,
            num_automaton_outputs,
            num_exp=experiment,
            log_dir=path + "/",
            dataset=dataset,
        )

        grounder.deepAutoma.float()
        grounder.deepAutoma.to(device)
        grounder.classifier.float()
        grounder.classifier.to(device)

        image_traj = []
        rew_traj = []
        info_traj = []
        sequence_accuracy = []
        image_accuracy = []

        positive_buffer = TraceReplayBuffer(capacity=50000)
        nonpos_buffer = TraceReplayBuffer(capacity=50000)

        pos_label = None
        if hasattr(env, "rew_dictionary") and 100 in env.rew_dictionary:
            pos_label = env.rew_dictionary[100]
        if pos_label is None:
            pos_label = 100

    optimizer = optim.Adam(params, lr=lr)
    episode_idx = 0
    all_mean_rewards = []
    all_mean_rewards_averaged = []

    # --- PPO Data Buffers ---
    ppo_states = []
    ppo_actions = []
    ppo_log_probs = []
    ppo_returns = []
    ppo_advantages = []

    for episode_idx in tqdm(range(max_episodes), desc="Training episodes", unit="episode"):
        episode_rewards = []
        done = False
        truncated = False

        obs, reward, info = env.reset()

        if method == "rm":
            state_dfa = torch.tensor(obs[0]).float().to(device)
            state_env = torch.tensor(obs[1]).float().to(device)

            if feature_extraction:
                state_env = cnn(state_env.view(-1, 3, 64, 64))
            state = torch.cat((state_env, state_dfa.unsqueeze(0)), 1).squeeze()
        else:
            state = torch.tensor(obs).float().to(device)

            if method == "rnn":
                h_0 = torch.zeros(num_layers, rnn_hidden_size).to(device).float()
                c_0 = torch.zeros(num_layers, rnn_hidden_size).to(device).float()
            elif method == "nrm":
                state_automa = np.zeros(num_of_states)
                state_automa[0] = 1.0
                state_automa = torch.tensor(state_automa).float().to(device)

            raw_state = state
            if feature_extraction:
                state = cnn(state.view(-1, 3, 64, 64))
                state = state.squeeze()

        if method == "rnn":
            out, (h_0, c_0) = rnn(state.unsqueeze(0), h_0, c_0)
            state = out
        elif method == "nrm":
            state_grounding = grounder.classifier(raw_state.unsqueeze(0))
            next_state_automa, reward_automa = grounder.deepAutoma.step(
                state_automa.unsqueeze(0), state_grounding, 1.0
            )
            state_automa = torch.zeros(num_of_states).to(device).float()
            state_automa[0] = 1.0
            state = torch.cat((state.unsqueeze(0), state_automa.unsqueeze(0)), dim=-1)
            state = state.squeeze()

        if method == "nrm":
            curr_traj = []
            curr_rew = []
            curr_info = []
            curr_traj.append(raw_state)
            curr_rew.append(reward)
            curr_info.append(info)

        while not (done or truncated):
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):
               
                # We need the state on the device for the ActorCritic
                state = torch.unsqueeze(state, 0).to(device)

                dist, value = model(state)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # --- PPO Buffer Appends ---
                # SAVE RAW INPUTS so we can recompute the CNN/RNN graph later!
                ppo_actions.append(action.detach())
                ppo_log_probs.append(log_prob.detach())
                
                if method == "rm":
                    # Save the raw environment image and dfa state
                    ppo_states.append((raw_state.detach(), state_dfa.detach())) 
                elif method == "nrm":
                    # Save raw image and current automa state
                    ppo_states.append((raw_state.detach(), state_automa.detach()))
                elif method == "rnn":
                    # Save raw image and previous hidden states
                    ppo_states.append((raw_state.detach(), h_0.detach(), c_0.detach()))
                else:
                    ppo_states.append(raw_state.detach())

                
                next_state, reward, done, truncated, info = env.step(action.item())

                if method == "rm":
                    state_dfa = torch.tensor(next_state[0]).float().to(device)
                    state_env = torch.tensor(next_state[1]).float().to(device)

                    if feature_extraction:
                        state_env = cnn(state_env.view(-1, 3, 64, 64))
                    next_state = torch.cat((state_env, state_dfa.unsqueeze(0)), 1).squeeze()
                else:
                    next_state = torch.tensor(next_state).float().to(device)
                    raw_state = next_state
                    if feature_extraction:
                        next_state = cnn(next_state.view(-1, 3, 64, 64))
                        next_state = next_state.squeeze()

                    if method == "rnn":
                        out, (h_0, c_0) = rnn(next_state.unsqueeze(0), h_0, c_0)
                        next_state = out
                    elif method == "nrm":
                        with torch.no_grad():
                            if reward != 0:
                                state_grounding = grounder.classifier(raw_state.unsqueeze(0))
                                next_state_automa, reward_automa = grounder.deepAutoma.step(
                                    state_automa.unsqueeze(0), state_grounding, 1.0
                                )
                                state_automa = next_state_automa.squeeze(0)
                                curr_traj.append(raw_state)
                                curr_rew.append(reward)
                                curr_info.append(info)
                            else:
                                next_state_automa = state_automa.unsqueeze(0)

                        next_state = torch.cat((next_state.unsqueeze(0), state_automa.unsqueeze(0)), dim=-1)
                        next_state = next_state.squeeze()

                state = next_state

                entropy += dist.entropy().mean()
                log_prob = torch.unsqueeze(log_prob, 0)
                log_probs.append(log_prob)
                values.append(value)
                
                reward = float(reward)
                episode_rewards.append(reward)
                reward_t = torch.tensor(reward).float().unsqueeze(0).unsqueeze(0)
                rewards.append(reward_t)

                formask = torch.tensor(1 if done else 0).float().unsqueeze(0).unsqueeze(0)
                masks.append(formask)

                if done or truncated:
                    break

            dist, next_value = model(next_state)
            returns = compute_returns(next_value, rewards, masks)

            values_cat = torch.cat(values)
            values_cat = values_cat.reshape((values_cat.size()[0], 1)).to(device)
            returns_cat = torch.cat(returns).to(device)
            
            advantage = returns_cat - values_cat

            # Append to PPO global buffers for the update phase
            ppo_returns.extend(returns_cat.detach())
            ppo_advantages.extend(advantage.detach())

        episode_idx += 1
        all_mean_rewards.append(np.sum(np.array(episode_rewards)))
        all_mean_rewards_averaged.append(mean(all_mean_rewards[-slide_wind:]))

        if method == "nrm":
            curr_traj_t = torch.stack(curr_traj).unsqueeze(0)
            curr_info_t = torch.LongTensor(curr_info).unsqueeze(0)

            acc, loss = eval_acceptance(
                grounder.classifier,
                grounder.deepAutoma,
                num_of_symbols,
                ([curr_traj_t], [curr_info_t]),
                automa_implementation="logic_circuit",
            )

            sequence_accuracy.append(acc)
            image_accuracy.append(0)

            with open(path + "/sequence_classification_accuracy_" + str(experiment) + ".txt", "a") as f:
                f.write("{}\n".format(acc))
            
            with open(path + "/image_classification_accuracy_" + str(experiment) + ".txt", "a") as f:
                f.write("{}\n".format(0))

            curr_traj = pad_list(curr_traj, max_traces + 1)
            curr_info = pad_list(curr_info, max_traces + 1)
            curr_rew = pad_list(curr_rew, max_traces + 1)

            if any(lbl == pos_label for lbl in curr_info):
                positive_buffer.add(curr_traj, curr_info)
            else:
                nonpos_buffer.add(curr_traj, curr_info)

       # --- PPO POLICY UPDATE ---
        if episode_idx % TT_policy == 0 and len(ppo_states) > 0:
            b_actions = torch.cat(ppo_actions).to(device)
            b_old_log_probs = torch.cat(ppo_log_probs).to(device)
            b_returns = torch.stack(ppo_returns).to(device)
            b_advantages = torch.stack(ppo_advantages).to(device)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            for _ in range(ppo_epochs):
                # 1. REBUILD THE COMPUTATION GRAPH
                recomputed_states = []
                
                for i in range(len(ppo_states)):
                    if method == "rm":
                        r_env, r_dfa = ppo_states[i]
                        if feature_extraction:
                            r_env = cnn(r_env.view(-1, 3, 64, 64))
                        s = torch.cat((r_env, r_dfa.unsqueeze(0)), 1).squeeze()
                        recomputed_states.append(s)
                        
                    elif method == "nrm":
                        r_raw, r_automa = ppo_states[i]
                        if feature_extraction:
                            r_raw = cnn(r_raw.view(-1, 3, 64, 64)).squeeze()
                        s = torch.cat((r_raw.unsqueeze(0), r_automa.unsqueeze(0)), dim=-1).squeeze()
                        recomputed_states.append(s)
                        
                    elif method == "rnn":
                        r_raw, r_h0, r_c0 = ppo_states[i]
                        if feature_extraction:
                            r_raw = cnn(r_raw.view(-1, 3, 64, 64)).squeeze()
                        out, _ = rnn(r_raw.unsqueeze(0).unsqueeze(0), r_h0, r_c0)
                        recomputed_states.append(out.squeeze())
                        
                    else:
                        r_raw = ppo_states[i]
                        if feature_extraction:
                            r_raw = cnn(r_raw.view(-1, 3, 64, 64)).squeeze()
                        recomputed_states.append(r_raw)

                # Stack the freshly computed states that now have gradients attached
                b_states_with_grad = torch.stack(recomputed_states).to(device)

                # 2. Forward pass through ActorCritic
                dist, new_values = model(b_states_with_grad)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                new_log_probs = new_log_probs.view(-1, 1)
                b_old_log_probs_view = b_old_log_probs.view(-1, 1)
                
                ratio = torch.exp(new_log_probs - b_old_log_probs_view)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (b_returns - new_values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()

            ppo_states.clear()
            ppo_actions.clear()
            ppo_log_probs.clear()
            ppo_returns.clear()
            ppo_advantages.clear()
        # Grounder Update
        if method == "nrm":
            if episode_idx % TT_grounder == 0:
                target_sample_size = target_batch_grounder
                total_data = len(positive_buffer) + len(nonpos_buffer)
                if total_data >= target_sample_size:
                    n_pos = target_sample_size // 2
                    n_neg = target_sample_size - n_pos

                    if len(positive_buffer) < n_pos:
                        pos_samples = positive_buffer.sample(len(positive_buffer))
                        n_neg += n_pos - len(positive_buffer)
                    else:
                        pos_samples = positive_buffer.sample(n_pos)

                    zero_samples = nonpos_buffer.sample(n_neg)
                    sampled = pos_samples + zero_samples

                    if len(sampled) >= 64:
                        bat_traj = [traj for (traj, labels) in sampled]
                        bat_labels = [labels for (traj, labels) in sampled]

                        grounder.set_dataset(bat_traj, bat_labels)
                        grounder.train_symbol_grounding(
                            grounder_epochs, batch_size=64, env=env
                        )

                image_traj = []
                rew_traj = []
                info_traj = []

        if episode_idx % TTT == 0 and len(all_mean_rewards) >= 100:
            plt.plot(
                [i for i in range(len(all_mean_rewards))], all_mean_rewards_averaged
            )
            plt.axhline(y=env.max_reward, color="r", linestyle="--")
            plt.xlabel("episode")
            plt.ylabel("mean episode rewards")
            plt.savefig(path + "/ImageEnvMeanRewardsReal_" + str(experiment) + ".png")
            plt.clf()
            plt.close()

            if method == "nrm":
                plt.plot([i for i in range(len(sequence_accuracy))], sequence_accuracy)
                plt.xlabel("episode")
                plt.ylabel("sequence classification accuracy")
                plt.savefig(
                    path + "/SequenceClassificationAccuracy_" + str(experiment) + ".png"
                )
                plt.clf()
                plt.close()

                plt.plot([i for i in range(len(image_accuracy))], image_accuracy)
                plt.xlabel("episode")
                plt.ylabel("image classification accuracy")
                plt.savefig(
                    path + "/ImageClassificationAccuracy_" + str(experiment) + ".png"
                )
                plt.clf()
                plt.close()

            plot_smoothed_sequence_classification_accuracy()

        ep_reward = all_mean_rewards[-1]
        f = open(path + "/train_rewards_" + str(experiment) + ".txt", "a")
        f.write(str(ep_reward) + "\n")
        f.close()
        
        if episode_idx % 100 == 0:
            print(
                "Mean cumulative reward in the last {} episodes: {}".format(
                    slide_wind, mean(all_mean_rewards[-slide_wind:])
                )
            )

        if len(all_mean_rewards) >= 100 and all_mean_rewards_averaged[-1] == 100:
            episode_idx = max_episodes

    if method == "nrm":
        with open(path + "/DeepAutoma_Before_Minimization_exp" + str(experiment) + ".pkl", 'wb') as outp:
            pickle.dump(grounder.deepAutoma, outp, pickle.HIGHEST_PROTOCOL)

            with open (path + "/buffer_exp" + str(experiment) + ".pkl", 'wb') as outp:
                pickle.dump({
                    "positive_buffer": positive_buffer,
                    "nonpos_buffer": nonpos_buffer
                }, outp, pickle.HIGHEST_PROTOCOL)

            with open(path + "/DeepAutoma_Symbol_Grounding_Classifier_exp" + str(experiment) + ".pkl", 'wb') as outp:
                pickle.dump(grounder.classifier, outp, pickle.HIGHEST_PROTOCOL)
                
    if feature_extraction:
        torch.save(
            cnn.state_dict(), path + "/cnn_state_dict_" + str(experiment) + ".pt"
        )