import csv
import os
import torch
import logging
import sys
import pickle


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name, storage_dir="storage"):
    return os.path.join(storage_dir, model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir, device):
    path = get_status_path(model_dir)
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    else:
        return None


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir, device):
    return get_status(model_dir, device)["vocab"]


def get_model_state(model_dir, device):
    return get_status(model_dir, device)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def close_txt_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def load_config(model_dir):
    path = os.path.join(model_dir, "config.pickle")
    if (not os.path.exists(path)):
        print(f"No config file found at: {path}")
    return pickle.load(open(path, "rb"))


def save_config(model_dir, config):
    path = os.path.join(model_dir, "config.pickle")
    create_folders_if_necessary(path)
    pickle.dump(config, open(path, "wb"))
