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

from stable_baselines3 import PPO


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
torch.autograd.set_detect_anomaly(True)


class TraceReplayBuffer_old:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def add(self, traj, labels, loss=0):
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


def recurrent_PPO(
    env,
    path,
    experiment,
    method,
    feature_extraction,
    num_of_states=None,
    num_of_symbols=None,
    hidden_size_rnn=50,
    formula_name="",
):

    model = PPO(
        "CnnPolicy", env, verbose=1, policy_kwargs={"normalize_images": False}
    )  
    model.learn(total_timesteps=10000)
