import argparse
import torch.multiprocessing as mp
import torch

from lab.train_agent_configs import *
from train_agent import train_agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    mp.set_start_method('spawn', force=True)
    train_agent(test_gridworld, device=device)