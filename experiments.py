from LTL_tasks import formulas
import absl.flags
import absl.app
import os
from RL.NRM.utils import set_seed
from RL.Env.Environment import GridWorldEnv
from RL.A2C import recurrent_A2C
from plot import plot

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
absl.flags.DEFINE_integer("NUM_EXPERIMENTS", 5, "num of runs for each test, default= 5")
absl.flags.DEFINE_integer("NUM_STATES", None, "num of states for the NRM, default= 30")
absl.flags.DEFINE_integer(
    "NUM_SYMBOLS", None, "num of symbols for the NRM, default= 30"
)


FLAGS = absl.flags.FLAGS


def launch_experiments(path, formula, experiment, env_type, method):
    set_seed(experiment)

    if env_type == "map_env":
        state_type = "symbolic"
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

        """
        RUNNARE MARIO con la minimizzazione con più stati e con la minimizzazione aggiustata (9 label finale e basta)
        
        """

    print(
        f"Experiment {experiment} on formula {formula[2]} with method {method} and state type {state_type}"
    )

    env = GridWorldEnv(
        formula,
        "human",
        state_type=state_type,
        use_dfa_state=use_dfa_state,
        train=False,
    )
    if not os.path.exists(path):
        os.makedirs(path)

    recurrent_A2C(
        env,
        path,
        experiment,
        method,
        feature_extraction,
        num_of_states=FLAGS.NUM_STATES,
        num_of_symbols=FLAGS.NUM_SYMBOLS,
    )


def main(argv):
    if not os.path.isdir(FLAGS.LOG_DIR):
        os.makedirs(FLAGS.LOG_DIR)
    for formula_idx, formula in enumerate(formulas):
        for experiment in range(FLAGS.NUM_EXPERIMENTS):
            print(f"Experiment {experiment} on formula {formula[2]}")
            path = FLAGS.LOG_DIR + str(formula[2])

            launch_experiments(path, formula, experiment, FLAGS.ENV, FLAGS.METHOD)
        plot(path, FLAGS.NUM_EXPERIMENTS, formula, 100)


if __name__ == "__main__":
    absl.app.run(main)
