import os
import argparse
import torch
import time

import utils


parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--model_dir", default="full_agent", type=str)
parser.add_argument("--ltl_sampler", default="Dataset_e54test_no-shuffle", type=str)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--formula_id", default=0, type=int)
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
print(f"\nConfig:\n{config}")

# load training status
status = utils.get_status(agent_dir, device)

# build environment
env = utils.make_env(
    config.env,
    progression_mode = config.progression_mode,
    ltl_sampler = args.ltl_sampler,
    seed = 1,
    intrinsic = config.int_reward,
    noLTL = config.noLTL,
    state_type = config.state_type,
    grounder = None,
    obs_size = config.obs_size
)
action_to_str = {0:"down", 1:"right", 2:"up", 3:"left"}

# set formula
env.sampler.sampled_tasks = args.formula_id

# create and load grounder
sym_grounder = utils.make_grounder(
    model_name = config.grounder_model,
    num_symbols = len(env.propositions),
    obs_size = config.obs_size,
    freeze_grounder = True
)
sym_grounder.load_state_dict(status["grounder_state"]) if sym_grounder is not None else None
sym_grounder.to(device) if sym_grounder is not None else None
env.env.sym_grounder = sym_grounder

agent = utils.Agent(env, env.observation_space, env.action_space, agent_dir, config.ignoreLTL, config.progression_mode,
                    config.gnn_model, config.recurrence, config.dumb_ac, device, False, 1, False)


# TEST

obs = env.reset()
done = False
step = 0

while not done:

    step += 1
    env.show()

    time.sleep(0.5)

    print(f"\n---")
    print(f"Step: {step}")
    print(f"Predicted Residual Task:")
    utils.pprint_ltl_formula(env.translate_formula(env.pred_ltl_goal))

    if env.real_ltl_goal != env.pred_ltl_goal:
        print("WRONG PREDICTED RESIDUAL FORMULA")

    print("\nAction: ", end="")

    action = agent.get_action(obs).item()
    print(action_to_str[action])

    obs, reward, done, info = env.step(action)

    real_sym = env.env.translate_formula(env.env.get_real_events())
    print(f"Real Symbol: {real_sym}")
    pred_sym = env.env.translate_formula(env.env.get_events())
    print(f"Pred Symbol: {pred_sym}")
    if env.env.get_events() != env.env.get_real_events():
        print("WRONG PREDICTION")
    print(f"Reward: {reward}")

    if done:
        break

env.show()
print("Done!")
print("Closing...")

time.sleep(2.0)
env.close()