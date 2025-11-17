import torch
import numpy as np
import os

import utils
from torch_ac import DictList
from replay_buffer import ReplayBuffer
from deep_automa import MultiTaskProbabilisticAutoma


# class for training the grounder
class GrounderAlgo():

    def __init__(self, grounder, env, train_grounder, max_env_steps=75, buffer_size=1024, lr=0.001, batch_size=32,
        update_steps=4, accumulation=1, evaluate_steps=1, use_early_stopping=False, patience=20, min_delta=0.0,
        save_dir=None, device=None):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.max_env_steps = max_env_steps
        self.buffer_size = buffer_size
        self.val_buffer_size = buffer_size // 4
        self.val_split_ratio = 0.2
        self.residual_exps = None

        self.batch_size = batch_size
        self.lr = lr
        self.update_steps = update_steps
        self.accumulation = accumulation
        self.evaluate_steps = evaluate_steps
        self.use_early_stopping = use_early_stopping

        self.grounder = grounder
        self.env = env
        self.sampler = env.sampler

        self.train_grounder = train_grounder and grounder is not None
        self.num_symbols = len(env.propositions) + 1

        if self.train_grounder:
            self.buffer = ReplayBuffer(capacity=self.buffer_size, device=device)
            self.val_buffer = ReplayBuffer(capacity=self.val_buffer_size, device=device)
            self.loss_func = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.grounder.parameters(), lr=lr)
            self.optimizer.zero_grad()
            assert self.sampler.has_automata

        else:
            self.buffer = None
            self.val_buffer = None
            self.loss_func = None
            self.optimizer = None

        self.patience = patience
        self.min_delta = min_delta
        self.save_path = os.path.join(save_dir, "grounder.pt") if save_dir else None
        self.early_stopping_counter = 0
        self.best_loss = float('inf')
        self.early_stop = False


    def add_episode(self, obss, rews, dfa_trans, dfa_rew):
        episode_count = self.buffer.total_episodes + self.val_buffer.total_episodes
        if episode_count % int(1 / self.val_split_ratio) == 0:
            self.val_buffer.push(obss, rews, dfa_trans, dfa_rew)
        else:
            self.buffer.push(obss, rews, dfa_trans, dfa_rew)


    def process_experiences(self, exps):

        if not self.train_grounder or self.early_stop:
            logs = {
                'buffer': 0, 'val_buffer': 0,
                'total_buffer': 0, 'total_val_buffer': 0
            }
            return logs

        # process last observations
        last_obs = {
            (obs['episode_id'], obs['env_id']): torch.tensor(obs['features'], device=self.device)
            for obs in exps.last_obs if obs is not None
        }

        # add residual experiences to exps
        exps = DictList({'obs': exps.obs, 'reward': exps.reward.long()})
        exps = utils.concat_dictlists(self.residual_exps, exps)

        used_mask = torch.zeros(exps.reward.shape, dtype=torch.bool)

        # compose finished episodes
        for episode_id, env_id in last_obs.keys():

            mask = (exps.obs.episode_id == episode_id) & (exps.obs.env_id == env_id)
            used_mask |= mask
            masked_exps = exps[mask]

            rew = masked_exps.reward[-1]
            frames = mask.sum() + 1

            to_add = (rew != 0 and frames <= self.max_env_steps+1) or (frames < self.max_env_steps+1)

            # add episode to the buffer
            if to_add:
                obss = torch.cat([masked_exps.obs.image, last_obs[episode_id, env_id].unsqueeze(0)], dim=0)
                rews = torch.cat([masked_exps.reward.new_zeros(1), masked_exps.reward], dim=0)
                task = masked_exps.obs.task_id[0]
                dfa = self.sampler.get_automaton(task)
                dfa_transitions = torch.tensor(dfa['transitions'], device=self.device, dtype=torch.int64)
                dfa_rewards = torch.tensor(dfa['rewards'], device=self.device, dtype=torch.int64)
                self.add_episode(obss, rews, dfa_transitions, dfa_rewards)

        self.residual_exps = DictList({
            'obs': exps.obs[~used_mask],
            'reward': exps.reward[~used_mask]
        })

        logs = {
            'buffer': len(self.buffer), 'val_buffer': len(self.val_buffer), 
            'total_buffer': self.buffer.total_episodes, 'total_val_buffer': self.val_buffer.total_episodes
        }

        return logs


    def collect_experiences(self, agent=None):

        if not self.train_grounder or self.early_stop:
            logs = {
                'buffer': 0, 'val_buffer': 0, 'total_buffer': 0, 'total_val_buffer': 0,
                'episode_return': 0.0, 'episode_frames': 0.0
                }
            return logs

        # reset the environment
        obs = self.env.reset()

        # agent starts in an empty cell (never terminates in 0 actions)
        done = False
        obss = [obs['features']]
        rews = [0]
        frames = 1

        # play the episode until termination
        while not done:
            action = agent.get_action(obs).item() if agent else self.env.action_space.sample()
            obs, rew, done, _ = self.env.step(action)
            obss.append(obs['features'])
            rews.append(rew)
            frames += 1
            done = done or frames >= self.max_env_steps+1

        to_add = (rew != 0 and frames <= self.max_env_steps+1) or (frames < self.max_env_steps+1)

        # add episode to the buffer
        if to_add:
            obss = torch.tensor(np.stack(obss), device=self.device, dtype=torch.float32)
            rews = torch.tensor(rews, device=self.device, dtype=torch.int64)
            dfa = self.env.sampler.get_current_automaton()
            dfa_transitions = torch.tensor(dfa['transitions'], device=self.device, dtype=torch.int64)
            dfa_rewards = torch.tensor(dfa['rewards'], device=self.device, dtype=torch.int64)
            self.add_episode(obss, rews, dfa_transitions, dfa_rewards)

        logs = {
            'buffer': len(self.buffer), 'val_buffer': len(self.val_buffer),
            'total_buffer': self.buffer.total_episodes, 'total_val_buffer': self.val_buffer.total_episodes,
            'episode_return': float(rew), 'episode_frames': float(len(rews))
        }

        return logs


    def update_parameters(self):

        if not self.train_grounder or len(self.buffer) == 0 or self.early_stop:
            logs = {
                'grounder_loss': 0.0, 'grounder_val_loss': 0.0, 
                'grounder_early_stop': self.early_stop
            }
            return logs

        losses = []
        val_losses = []

        # reset gradient
        self.optimizer.zero_grad()

        # train
        for update_id in range(self.update_steps):

            # sample from the buffer
            batch_size = min(self.batch_size, len(self.buffer))
            obss, rews, dfa_trans, dfa_rew = self.buffer.sample(batch_size)

            # build the differentiable reward machine for the task
            deepDFA = MultiTaskProbabilisticAutoma(
                batch_size = batch_size,
                num_actions = self.num_symbols,
                num_states = max([tr.shape[0] for tr in dfa_trans]),
                reward_type = "ternary",
                device = self.device
            )
            deepDFA.initFromDfas(dfa_trans, dfa_rew)

            # obtain probability of symbols from observations
            symbols_probs = self.grounder(obss.view(-1, *obss.shape[2:]))
            symbols_probs = symbols_probs.view(*obss.shape[:2], -1)

            # predict reward probability from symbols probability with DeepDFA
            _, preds = deepDFA(symbols_probs)
            preds = preds.view(-1, deepDFA.num_rewards)

            # maps rewards to label
            labels = (rews + 1).view(-1)

            # compute and backpropagate loss
            loss = self.loss_func(preds, labels)
            (loss / self.accumulation).backward()
            losses.append(loss.item())

            # update self.grounder
            if (update_id + 1) % self.accumulation == 0 or (update_id + 1) == self.update_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = sum(losses) / self.update_steps

        # validation
        with torch.no_grad():

            for batch in self.val_buffer.iter_batches(batch_size=self.batch_size):

                obss, rews, dfa_trans, dfa_rew = batch
                batch_size = obss.shape[0]

                deepDFA = MultiTaskProbabilisticAutoma(
                    batch_size = batch_size,
                    num_actions = self.num_symbols,
                    num_states = max([tr.shape[0] for tr in dfa_trans]),
                    reward_type = "ternary",
                    device = self.device
                )
                deepDFA.initFromDfas(dfa_trans, dfa_rew)

                symbols_probs = self.grounder(obss.view(-1, *obss.shape[2:]))
                symbols_probs = symbols_probs.view(*obss.shape[:2], -1)

                _, preds = deepDFA(symbols_probs)
                preds = preds.view(-1, deepDFA.num_rewards)

                labels = (rews + 1).view(-1)

                loss = self.loss_func(preds, labels)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / max(1, len(val_losses))

        # early stopping
        self.early_stopping_check(avg_val_loss)

        # log some values
        logs = {
            'grounder_loss': avg_loss, 'grounder_val_loss': avg_val_loss,
            'grounder_early_stop': self.early_stop
        }

        return logs


    def evaluate(self):

        if not self.train_grounder:
            logs = {
                'grounder_acc': 0.0,
                'grounder_recall': [0.0 for _ in range(self.num_symbols)]
            }
            return logs

        coords = self.env.env.loc_to_label.keys()

        real_syms = []
        pred_syms = []

        # accumulate real and pred symbols
        with torch.no_grad():

            for _ in range(self.evaluate_steps):

                self.env.reset()

                step_real_syms = [self.env.env.loc_to_label[(r, c)] for (r, c) in coords]
                step_real_syms = torch.tensor(step_real_syms, device=self.device, dtype=torch.int32)
                real_syms.append(step_real_syms)

                images = np.stack([self.env.env.loc_to_obs[(r, c)] for (r, c) in coords])
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                step_pred_syms = torch.argmax(self.grounder(images), dim=-1)
                pred_syms.append(step_pred_syms)

        real_syms = torch.cat(real_syms, dim=0)
        pred_syms = torch.cat(pred_syms, dim=0)

        # compute accuracy
        correct = (pred_syms == real_syms)
        acc = correct.float().mean()

        # compute recall
        true_pos = torch.zeros(self.num_symbols, device=self.device)
        false_neg = torch.zeros(self.num_symbols, device=self.device)
        for sym in range(self.num_symbols):
            true_pos[sym] = torch.sum((pred_syms == sym) & (real_syms == sym))
            false_neg[sym] = torch.sum((pred_syms != sym) & (real_syms == sym))
        recall = true_pos / (true_pos + false_neg + 1e-8)

        # log some values
        logs = {
            'grounder_acc': acc.item(),
            'grounder_recall': recall.tolist()
        }

        return logs


    def early_stopping_check(self, loss):

        if not self.use_early_stopping:
            return

        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.save_grounder()
            self.early_stopping_counter = 0

        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                self.early_stop = True
                self.load_grounder()


    def save_grounder(self):
        torch.save(self.grounder.state_dict(), self.save_path)


    def load_grounder(self):
        self.grounder.load_state_dict(torch.load(self.save_path))


    def clear_buffers(self):
        if self.train_grounder:
            self.buffer.clear()
            self.val_buffer.clear()