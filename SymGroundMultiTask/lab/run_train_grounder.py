import argparse
import torch

from lab.train_grounder_configs import *
from train_grounder import train_grounder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_grounder(train_grounder_base, device=device)