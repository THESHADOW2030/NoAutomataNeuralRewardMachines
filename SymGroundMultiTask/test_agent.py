import os
import argparse
import torch
import time

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--model_dir', default='full_agent', type=str)
parser.add_argument('--ltl_sampler', default='Dataset_e54test_no-shuffle', type=str)
parser.add_argument('--argmax', dest='argmax', default=True, action='store_true')
parser.add_argument('--no-argmax', dest='argmax', action='store_false')
parser.add_argument('--eval_procs', default=1, type=int)
parser.add_argument('--eval_episodes', default=1000, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--max_num_steps', default=150, type=int)
args = parser.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device)

utils.set_seed(args.seed)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# compute agent dir
storage_dir = os.path.join(REPO_DIR, "storage")
agent_dir = os.path.join(storage_dir, args.model_dir)

# load training config
config = utils.load_config(agent_dir)

# load training status
status = utils.get_status(agent_dir, device)

print("\n---\n")
print("Config:")
for field_name, value in vars(config).items():
    print(f"\t{field_name}: {value}")
print(f"\nDevice: {device}")
print("\n---\n")

# create evaluator
evalu = utils.Eval(config.eval_env, agent_dir, args.ltl_sampler, args.seed, device, config.state_type, None,
                   config.obs_size, args.argmax, args.eval_procs, config.ignoreLTL, config.progression_mode, config.gnn_model,
                   config.recurrence, config.dumb_ac, args.max_num_steps)

# create and load grounder
sym_grounder = utils.make_grounder(
    model_name = config.grounder_model,
    num_symbols = len(evalu.eval_env.envs[0].propositions) + 1,
    obs_size = config.obs_size,
    freeze_grounder = True
)
sym_grounder.load_state_dict(status["grounder_state"]) if sym_grounder is not None else None
sym_grounder.to(device) if sym_grounder is not None else None
for env in evalu.eval_env.envs:
    env.env.sym_grounder = sym_grounder


# TEST

print("Starting Evaluation...")

eval_start_time = time.time()
return_per_episode, frames_per_episode = evalu.eval(args.eval_episodes)
eval_end_time = time.time()

duration = int(eval_end_time - eval_start_time)

total_eval_frames = sum(frames_per_episode)
average_discounted_return = utils.average_discounted_return(return_per_episode, frames_per_episode, config.discount)
return_per_episode = utils.synthesize(return_per_episode)
frames_per_episode = utils.synthesize(frames_per_episode)

header = ['time/frames', 'time/duration']
data = [total_eval_frames, duration]
header += ['return/' + key for key in return_per_episode.keys()]
data += return_per_episode.values()
header += ['average_discounted_return']
data += [average_discounted_return]
header += ['episode_frames/' + key for key in frames_per_episode.keys()]
data += frames_per_episode.values()

print(f"Evaluator {evalu.eval_name}")
print(
    ("F {:7.0f} | D {:5} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | ADR {:.3f}" +
    " | F:μσmM {:4.1f} {:4.1f} {:2.0f} {:2.0f}").format(*data)
)

# kill subprocesses
evalu.eval_env.close()