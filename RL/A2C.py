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
import numpy as np
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
torch.autograd.set_detect_anomaly(True)

import os
import pickle


# max number of episodes
max_episodes = 10000  # 10000


# output of rnn
rnn_outputs = 5

# layers of rnn
num_layers = 2

# hyper params:
hidden_size = 120  # of a2c
rnn_hidden_size = 50  # of rnn

# slidind window
slide_wind = 100

lr = 0.0004

# we train the policy every num_steps
num_steps = 5
TT_policy = 5
TT_grounder = 100
grounder_epochs = 100

# we plot the graph every TTT episode
TTT = 10

# --- OPTIMIZED HYPERPARAMS ---

# 1. Update Policy frequently to fix the Critic quickly
TT_policy = 1  # Was 5. Update every episode.

# 2. Train Grounder frequently so the Agent adapts to changes
TT_grounder = 10  # Was 50. Don't let the agent run blind for too long.

# 3. Prevent Overfitting in the Grounder
grounder_epochs = 150  # Was 100. Short bursts of learning are better.

# 4. A2C Params (Standard)
lr = 0.0007  # Slightly higher because batch size is effectively smaller now
num_steps = 5  # Keep this (standard n-step return)
hidden_size = 120  # Keep this

# 5. Data Collection
# Ensure your buffer logic doesn't crash if you don't have 64 samples yet
# You might want to lower this to 32 if episodes are successful rarely
target_batch_grounder = 1024


# -------------------GEMINI-------------------#
TT_grounder = 10
grounder_epochs = 10
max_episodes = 10000


import os


def plot_smoothed_sequence_classification_accuracy():
    # read sequence_classification_accuracy_0.txt
    for i in range(5):
        # check if file exists
        if not os.path.exists(
            os.path.join(
                os.path.dirname(__file__), f"sequence_classification_accuracy_{i}.txt"
            )
        ):
            continue
        with open(
            os.path.join(
                os.path.dirname(__file__), f"sequence_classification_accuracy_{i}.txt"
            ),
            "r",
        ) as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()]
        lines = list(map(float, lines))
        smoothed = []
        mean = 0
        for i, line in enumerate(lines):
            if i % 100 == 0:
                smoothed.append((i, mean / 100))
                mean = 0
            mean += line
            if i == len(lines) - 1:
                smoothed.append((i, mean / (i % 100 + 1)))  # handle last segment
        lines = smoothed

        # plot the smoothed values
        import matplotlib.pyplot as plt

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

        print(list(lines))


# Compute the returns (of the rewards) for one episode
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
):

    # recurrency =
    #       - 'rnn'     (rnn+A2C)
    #       - 'nrm'     (grounding+A2C)
    #       - 'rm'    (reward machines)

    #################### reinitialize files if they exist or create them

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

    print(
        f"num_of_states: {num_of_states}, num_of_symbols: {num_of_symbols}, num_automaton_outputs: {num_automaton_outputs}"
    )

    saved_traces = []

    # number of actions
    num_outputs = env.action_space.n

    params = []
    # --- MODEL INITIALIZATION ---
    if feature_extraction:
        cnn = Net().to(device)
        cnn.float()  # Model is Float32
        CNN_output_size = 16
        num_inputs = CNN_output_size
        params += list(cnn.parameters())
    else:
        num_inputs = env.state_space_size

    if method == "rnn":
        model = ActorCritic(rnn_hidden_size, num_outputs, hidden_size).to(device)
    else:
        print(
            f"PASSING num_inputs: {num_inputs}, num_outputs: {num_outputs}, hidden_size: {hidden_size}"
        )
        model = ActorCritic(num_inputs + num_of_states, num_outputs, hidden_size).to(
            device
        )

    params += list(model.parameters())
    model.float()  # Model is Float32

    if method == "rnn":
        rnn = RNN(num_inputs, rnn_hidden_size, num_layers).to(device)
        rnn.float()  # Model is Float32
        params += list(rnn.parameters())
    elif method == "nrm":
        # ... file creation ...
        f = open(
            path + "/sequence_classification_accuracy_" + str(experiment) + ".txt", "w"
        )
        f.close()
        f = open(
            path + "/image_classification_accuracy_" + str(experiment) + ".txt", "w"
        )
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

        # ### FIX: Initialize Grounder as Float
        grounder.deepAutoma.float()
        grounder.deepAutoma.to(device)
        grounder.classifier.float()
        grounder.classifier.to(device)

        # ... (buffer initialization code remains the same) ...
        image_traj = []
        rew_traj = []
        info_traj = []
        sequence_accuracy = []
        image_accuracy = []

        class TraceReplayBuffer:
            def __init__(self, capacity=2000):
                self.capacity = capacity
                self.data = deque(maxlen=capacity)

            def add(self, traj, labels):
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

        positive_buffer = TraceReplayBuffer(capacity=50000)
        nonpos_buffer = TraceReplayBuffer(capacity=50000)

        pos_label = None
        if hasattr(env, "rew_dictionary") and 100 in env.rew_dictionary:
            pos_label = env.rew_dictionary[100]
        if pos_label is None:
            pos_label = 100

    optimizer = optim.Adam(params, lr=lr)
    episode_idx = 0
    advantage_cat = torch.tensor([]).to(device)
    log_probs_cat = torch.tensor([]).to(device)
    all_mean_rewards = []
    all_mean_rewards_averaged = []

    for episode_idx in tqdm(
        range(max_episodes), desc="Training episodes", unit="episode"
    ):
        episode_rewards = []
        done = False
        truncated = False

        obs, reward, info = env.reset()

        # --- 1. STATE INITIALIZATION FIX ---
        if method == "rm":
            # ### FIX: Convert to float() immediately
            state_dfa = torch.tensor(obs[0]).float().to(device)
            state_env = torch.tensor(obs[1]).float().to(device)

            if feature_extraction:
                state_env = cnn(state_env.view(-1, 3, 64, 64))

            state = torch.cat((state_env, state_dfa.unsqueeze(0)), 1).squeeze()
        else:
            # ### FIX: Use .float() instead of .DoubleTensor()
            # This fixes the mismatch with the CNN (which is float)
            state = torch.tensor(obs).float().to(device)

            if method == "rnn":
                h_0 = torch.zeros(num_layers, rnn_hidden_size).to(device).float()
                c_0 = torch.zeros(num_layers, rnn_hidden_size).to(device).float()
            elif method == "nrm":
                state_automa = np.zeros(num_of_states)
                state_automa[0] = 1.0
                # ### FIX: Automa state must be float for concatenation
                state_automa = torch.tensor(state_automa).float().to(device)

            raw_state = state
            if feature_extraction:
                # Now state is float and cnn is float -> NO ERROR
                state = cnn(state.view(-1, 3, 64, 64))
                state = state.squeeze()

        # first step with RNN or dfa
        if method == "rnn":
            out, (h_0, c_0) = rnn(state.unsqueeze(0), h_0, c_0)
            state = out
        elif method == "nrm":
            # Inputs are already float, so this works
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
                state = torch.unsqueeze(state, 0)
                state = state.to(device)

                dist, value = model(state)
                action = dist.sample()

                next_state, reward, done, truncated, info = env.step(action.item())

                # --- 2. NEXT STATE FIX ---
                if method == "rm":
                    # ### FIX: Float
                    state_dfa = torch.tensor(next_state[0]).float().to(device)
                    state_env = torch.tensor(next_state[1]).float().to(device)

                    if feature_extraction:
                        state_env = cnn(state_env.view(-1, 3, 64, 64))
                    next_state = torch.cat(
                        (state_env, state_dfa.unsqueeze(0)), 1
                    ).squeeze()
                else:
                    # ### FIX: Use .float() instead of .DoubleTensor()
                    next_state = torch.tensor(next_state).float().to(device)

                    raw_state = next_state
                    if feature_extraction:
                        # Now next_state is float and cnn is float -> NO ERROR
                        next_state = cnn(next_state.view(-1, 3, 64, 64))
                        next_state = next_state.squeeze()

                    if method == "rnn":
                        out, (h_0, c_0) = rnn(next_state.unsqueeze(0), h_0, c_0)
                        next_state = out
                    elif method == "nrm":
                        with torch.no_grad():
                            if reward != 0:
                                # Input is already float
                                state_grounding = grounder.classifier(
                                    raw_state.unsqueeze(0)
                                )
                                next_state_automa, reward_automa = (
                                    grounder.deepAutoma.step(
                                        state_automa.unsqueeze(0), state_grounding, 1.0
                                    )
                                )
                                state_automa = next_state_automa.squeeze(0)
                                curr_traj.append(raw_state)
                                curr_rew.append(reward)
                                curr_info.append(info)
                            else:
                                next_state_automa = state_automa.unsqueeze(0)

                        next_state = torch.cat(
                            (next_state.unsqueeze(0), state_automa.unsqueeze(0)), dim=-1
                        )
                        next_state = next_state.squeeze()

                state = next_state

                # ... (rest of the loop remains the same) ...
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_prob = torch.unsqueeze(log_prob, 0)
                log_probs.append(log_prob)
                values.append(value)
                reward = float(reward)
                episode_rewards.append(reward)

                # Rewards processing
                reward = (
                    torch.tensor(reward).float().unsqueeze(0).unsqueeze(0)
                )  # Ensure float
                rewards.append(reward)

                formask = (
                    torch.tensor(1 if done else 0).float().unsqueeze(0).unsqueeze(0)
                )  # Ensure float
                masks.append(formask)

                if done or truncated:
                    break

            # ... (Update logic, Actor Critic Loss) ...
            dist, next_value = model(next_state)
            returns = compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns)
            values = torch.cat(values)
            values = values.reshape((values.size()[0], 1))

            # Ensure everything is on device
            log_probs = log_probs.to(device)
            returns = returns.to(device)
            advantage = returns - values

            log_probs_cat = torch.cat((log_probs_cat, log_probs), 0)
            advantage_cat = torch.cat((advantage_cat, advantage), 0)

            torch.cuda.empty_cache()

        # ... (Episode end logic, grounder training, plotting) ...
        # (This part of your code is fine, just ensure grounder.train_symbol_grounding is called correctly)
        episode_idx += 1
        all_mean_rewards.append(np.sum(np.array(episode_rewards)))
        all_mean_rewards_averaged.append(mean(all_mean_rewards[-slide_wind:]))

        if method == "nrm":
            # ... accuracy calculation ...
            curr_traj_t = torch.stack(curr_traj).unsqueeze(0)
            curr_info_t = torch.LongTensor(curr_info).unsqueeze(0)

            # Use logic_circuit for speed
            acc = eval_acceptance(
                grounder.classifier,
                grounder.deepAutoma,
                num_of_symbols,
                ([curr_traj_t], [curr_info_t]),
                automa_implementation="logic_circuit",
            )

            # ... appending to files ...
            sequence_accuracy.append(acc)
            # ... image accuracy ...
            image_accuracy.append(0)  # placeholder

            # --- INSERT THIS CODE HERE ---
            with open(path + "/sequence_classification_accuracy_" + str(experiment) + ".txt", "a") as f:
                f.write("{}\n".format(acc))
            
            with open(path + "/image_classification_accuracy_" + str(experiment) + ".txt", "a") as f:
                f.write("{}\n".format(0))

            # --- FIX STARTS HERE ---
            
            # 1. Pad the Trajectory (You already had this)
            curr_traj = pad_list(curr_traj, env.max_num_steps + 1)
            
            # 2. CRITICAL: Pad the Info/Labels (You were missing this)
            # Without this, episodes of different lengths create a ragged list,
            # causing the "expected sequence of length..." error in torch.LongTensor
            curr_info = pad_list(curr_info, env.max_num_steps + 1)
            
            # 3. Optional: Pad rewards if used elsewhere, to maintain consistency
            curr_rew = pad_list(curr_rew, env.max_num_steps + 1)

            # --- FIX ENDS HERE ---

            # Add to buffers
            if any(lbl == pos_label for lbl in curr_info):
                positive_buffer.add(curr_traj, curr_info)
            else:
                nonpos_buffer.add(curr_traj, curr_info)

        # Policy Update
        if episode_idx % TT_policy == 0:
            log_probs_cat = torch.unsqueeze(log_probs_cat, dim=1)
            actor_loss = -(log_probs_cat * advantage_cat).mean()
            critic_loss = advantage_cat.pow(2).mean()
            loss = 0.3 * actor_loss + 0.5 * critic_loss - 0.0001 * entropy

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            log_probs_cat = torch.tensor([]).to(device)
            advantage_cat = torch.tensor([]).to(device)

        # Grounder Update
        if method == "nrm":
            print(
                f"Episode {episode_idx}, TT_policy {TT_policy}, TT_grounder {TT_grounder}, sequence accuracy (last) {sequence_accuracy[-1]}"
            )
            if episode_idx % TT_grounder == 0:
                # ... Sampling logic ...
                target_sample_size = target_batch_grounder
                total_data = len(positive_buffer) + len(nonpos_buffer)
                print(f"LOG: total_data {total_data}")
                if total_data >= target_sample_size:
                    # Balanced Sampling
                    n_pos = target_sample_size // 2
                    n_neg = target_sample_size - n_pos

                    if len(positive_buffer) < n_pos:
                        pos_samples = positive_buffer.sample(len(positive_buffer))
                        n_neg += n_pos - len(positive_buffer)
                    else:
                        pos_samples = positive_buffer.sample(n_pos)

                    zero_samples = nonpos_buffer.sample(n_neg)
                    sampled = pos_samples + zero_samples
                    print(f"LOG: len(sampled) {len(sampled)}")

                    if len(sampled) >= 64:
                        bat_traj = [traj for (traj, labels) in sampled]
                        bat_labels = [labels for (traj, labels) in sampled]

                        grounder.set_dataset(bat_traj, bat_labels)
                        # TRAIN
                        grounder.train_symbol_grounding(
                            grounder_epochs, batch_size=64, env=env
                        )

                image_traj = []
                rew_traj = []
                info_traj = []
                sum_rew_traj = []
                #grounder.eval_symbol_grounding(env=env)

        if episode_idx % TTT == 0 and len(all_mean_rewards) >= 100:
            ## plot rewards
            plt.plot(
                [i for i in range(len(all_mean_rewards))], all_mean_rewards_averaged
            )
            plt.axhline(y=env.max_reward, color="r", linestyle="--")
            plt.xlabel("episode")
            plt.ylabel("mean episode rewards")
            plt.savefig(path + "/ImageEnvMeanRewardsReal_" + str(experiment) + ".png")
            plt.clf()
            plt.close()

            # plot the accuracies for nrm
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

        # else:
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

        


        # exit()
    dfa = grounder.deepAutoma.net2dfa(min_temp=0.00001, name_automata=path + "/" + "final_dfa_exp" + str(experiment))
    dfa.to_graphviz().render(path + "/" + "final_dfa_exp" + str(experiment) + ".dot")
    with open(path + "/" + "final_dfa_exp" + str(experiment) + ".pkl", 'wb') as outp:
        pickle.dump(dfa, outp, pickle.HIGHEST_PROTOCOL)
    # save the cnn state dict in case of feature extraction
    if feature_extraction:
        torch.save(
            cnn.state_dict(), path + "/cnn_state_dict_" + str(experiment) + ".pt"
        )

        # save for future use in a pickle and in case in exists, append it
    os.makedirs(path + "/traces/", exist_ok=True)  # Create it if it doesn't exist
    with open(f"{path}/traces/exp{str(experiment)}.pkl", "wb") as outp:
        print(f"Saving the traces in {path}/traces/exp{str(experiment)}.pkl")
        # pickle.dump(saved_traces, outp, pickle.HIGHEST_PROTOCOL)
