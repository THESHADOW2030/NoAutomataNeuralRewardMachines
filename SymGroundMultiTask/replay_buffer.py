from collections import deque
import torch
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=1000, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.total_episodes = 0
        self.buffer = deque(maxlen=capacity)


    def push(self, obss, rews, dfa_trans, dfa_rews):
        self.buffer.append((obss, rews, dfa_trans, dfa_rews))
        self.total_episodes += 1


    def __len__(self):
        return len(self.buffer)


    def __iter__(self):
        for obss, rews, dfa_trans, dfa_rews in self.buffer:
            obss = obss.to(self.device)
            rews = rews.to(self.device)
            dfa_trans = dfa_trans.to(self.device)
            dfa_rews = dfa_rews.to(self.device)
            yield obss, rews, dfa_trans, dfa_rews


    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        obss, rews, dfa_trans, dfa_rews = zip(*batch)
        obss = self._pad_batch(obss).to(self.device)
        rews = self._pad_batch(rews).to(self.device)
        dfa_trans = [tr.to(self.device) for tr in dfa_trans]
        dfa_rews = [rw.to(self.device) for rw in dfa_rews]
        return obss, rews, dfa_trans, dfa_rews


    def _pad_batch(self, sequences):
        max_len = max(seq.shape[0] for seq in sequences)
        padded = []
        for seq in sequences:
            seq = self._pad_repeat_last(seq, max_len)
            padded.append(seq)
        return torch.stack(padded)


    def _pad_repeat_last(self, seq, length):
        pad_len = length - seq.shape[0]
        if pad_len > 0:
            repeat = seq[-1:].expand(pad_len, *seq.shape[1:])
            seq = torch.cat([seq, repeat], dim=0)
        return seq


    def iter_batches(self, batch_size):
        buffer_list = list(self.buffer)
        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i:i + batch_size]
            obss, rews, dfa_trans, dfa_rews = zip(*batch)
            obss = self._pad_batch(obss).to(self.device)
            rews = self._pad_batch(rews).to(self.device)
            dfa_trans = [tr.to(self.device) for tr in dfa_trans]
            dfa_rews = [rw.to(self.device) for rw in dfa_rews]
            yield obss, rews, dfa_trans, dfa_rews


    def clear(self):
        self.buffer.clear()