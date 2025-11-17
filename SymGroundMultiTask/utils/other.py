import numpy as np
import torch
import collections


from torch_ac import DictList


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d


def average_reward_per_step(returns, num_frames):
    avgs = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        avgs.append(returns[i] / num_frames[i])

    return np.mean(avgs)


def average_discounted_return(returns, num_frames, disc):
    discounted_returns = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        discounted_returns.append(returns[i] * (disc ** (num_frames[i]-1)))

    return np.mean(discounted_returns)


def empty_episode_logs():
    logs = {
        "return_per_episode": [],
        "reshaped_return_per_episode": [],
        "num_frames_per_episode": [],
        "num_frames": 0
    }
    return logs


def empty_buffer_logs():
    logs = {
        'buffer': 0, 'val_buffer': 0, 
        'total_buffer': 0, 'total_val_buffer': 0
    }
    return logs


def empty_algo_logs():
    logs = {
        "entropy": 0.0,
        "value": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "grad_norm": 0.0
    }
    return logs


def empty_grounder_algo_logs():
    logs = {
        'grounder_loss': 0.0,
        "grounder_val_loss": 0.0
    }
    return logs


def empty_grounder_eval_logs(n_syms):
    logs = {
        'grounder_acc': 0.0,
        'grounder_recall': [0.0] * n_syms
    }
    return logs


def accumulate_episode_logs(logs_exp, logs):

    if logs.keys() == logs_exp.keys():
        logs_exp["return_per_episode"] += logs["return_per_episode"]
        logs_exp["reshaped_return_per_episode"] += logs["reshaped_return_per_episode"]
        logs_exp["num_frames_per_episode"] += logs["num_frames_per_episode"]
        logs_exp["num_frames"] += logs["num_frames"]

    else:
        logs_exp["return_per_episode"].append(logs["episode_return"])
        logs_exp["reshaped_return_per_episode"].append(logs["episode_return"])
        logs_exp["num_frames_per_episode"].append(logs["episode_frames"])
        logs_exp["num_frames"] += logs["episode_frames"]

    return logs_exp


def elaborate_episode_logs(logs, discount):

    return_per_episode = synthesize(logs["return_per_episode"])
    rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"])
    avg_reward_per_step = average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
    avg_discounted_return = average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], discount)
    num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

    episode_logs = {
        "return_per_episode": return_per_episode,
        "rreturn_per_episode": rreturn_per_episode,
        "average_reward_per_step": avg_reward_per_step,
        "average_discounted_return": avg_discounted_return,
        "num_frames_per_episode": num_frames_per_episode,
        "num_frames": logs["num_frames"]
    }

    return episode_logs


def concat_dictlists(dl1, dl2):

    if dl1 == None:
        return dl2
    if dl2 == None:
        return dl1

    if not isinstance(dl1, DictList) or not isinstance(dl2, DictList):
        raise TypeError("Both arguments must be instances of DictList")
    if set(dl1.keys()) != set(dl2.keys()):
        raise ValueError("Both DictList instances must have the same keys")

    result = {}
    for k in dl1:
        v1 = dict.__getitem__(dl1, k)
        v2 = dict.__getitem__(dl2, k)

        if isinstance(v1, DictList) and isinstance(v2, DictList):
            result[k] = concat_dictlists(v1, v2)
        elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            result[k] = torch.cat([v1, v2], dim=0)
        elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            result[k] = np.concatenate([v1, v2], axis=0)
        else:
            result[k] = v1 + v2

    return DictList(result)