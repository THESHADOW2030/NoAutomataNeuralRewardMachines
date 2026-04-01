from LTL_tasks import formulas
import absl.flags
import absl.app
import os
from RL.Env.SB3_wrapper import SB3CompatibilityWrapper
from RL.NRM.utils import set_seed
from RL.Env.Environment_update import GridWorldEnv
from RL.A2C import recurrent_A2C
from plot import plot
from RL.recurrent_PPO import recurrent_PPO

import random


# flags
absl.flags.DEFINE_string(
    "METHOD", "nrm", "Method to test, one in ['rnn', 'nrm', 'rm'], default= 'rnn' "
)
absl.flags.DEFINE_string(
    "ENV",
    "image_env",
    "Environment to test, one in ['map_env', 'image_env'], default= 'map_env' ",
)
absl.flags.DEFINE_string(
    "LOG_DIR", "Results/", "path where to save the results, default='Results/'"
)
absl.flags.DEFINE_integer("NUM_EXPERIMENTS", 4, "num of runs for each test, default= 5")
absl.flags.DEFINE_integer("NUM_STATES", None, "num of states for the NRM, default= 30")
absl.flags.DEFINE_integer("NUM_SYMBOLS", None, "num of symbols for the NRM, default= 5")
absl.flags.DEFINE_integer("NUM_HIDDEN_SIZE_RNN", 50, "hidden size for the RNN, default= 50")



FLAGS = absl.flags.FLAGS


def launch_experiments(path, formula, experiment, env_type, method):
    set_seed(experiment)

    if env_type == "map_env":
        state_type = "symbol"
        feature_extraction = False
    elif env_type == "image_env":
        state_type = "image"
        feature_extraction = True

    if method == "rnn":
        use_dfa_state = False
    elif method == "nrm":
        use_dfa_state = False
    elif method == "rm":
        use_dfa_state = True

    env = GridWorldEnv(
        formula,
        "human",
        state_type=state_type,
        use_dfa_state=use_dfa_state,
        train=False,
    )

    env = SB3CompatibilityWrapper(env)


    
    
    if not os.path.exists(path):
         os.makedirs(path)
   
    # exit()

    recurrent_PPO(
        env,
        path,
        experiment,
        method,
        feature_extraction,
        num_of_states=FLAGS.NUM_STATES,
        num_of_symbols=FLAGS.NUM_SYMBOLS,
        hidden_size_rnn=FLAGS.NUM_HIDDEN_SIZE_RNN,
        formula_name=formula[2],

    )


def main(argv):

    if not os.path.isdir(FLAGS.LOG_DIR):
        os.makedirs(FLAGS.LOG_DIR)
    for experiment in range(1, FLAGS.NUM_EXPERIMENTS + 1):
        for formula_idx, formula in enumerate(formulas):
        
            print(f"Experiment {experiment} on formula {formula[2]}")
            path = FLAGS.LOG_DIR + str(formula[2]) + f"/{FLAGS.METHOD}_{FLAGS.ENV}" + f"/NUM_STATES_{FLAGS.NUM_STATES}_NUM_SYMBOLS_{FLAGS.NUM_SYMBOLS}" + f"/exp{experiment}"

            launch_experiments(path, formula, experiment, FLAGS.ENV, FLAGS.METHOD)
        plot(path, FLAGS.NUM_EXPERIMENTS, formula, 100)


if __name__ == "__main__":
    absl.app.run(main)
