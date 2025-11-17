import argparse
import torch

from lab.ltl_bootcamp_configs import *
from train_agent import train_agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_agent(test_simple_ltl_5l, device=device)