from pathlib import Path
import os
from .dataset import Dataset


DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


# Available datasets

e54_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_1_5_1_4",
    allow_duplicates = False,
    disjoint_from = None,
)

e54dup_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54dup")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_1_5_1_4",
    allow_duplicates = True,
    disjoint_from = None,
)

e54test_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54test")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_1_5_1_4",
    allow_duplicates = False,
    disjoint_from = e54_dataset,
)

e54duptest_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54duptest")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_1_5_1_4",
    allow_duplicates = True,
    disjoint_from = e54dup_dataset,
)


e54prop12dup_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54prop12dup")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
    sampler = "Eventually_1_5_1_4",
    allow_duplicates = True,
    disjoint_from = None,
)

e54prop12duptest_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54prop12duptest")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
    sampler = "Eventually_1_5_1_4",
    allow_duplicates = True,
    disjoint_from = e54prop12dup_dataset,
)


ga321_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga321")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_1_3_1_2_1_1",
    allow_duplicates = False,
    disjoint_from = None,
)

ga321dup_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga321dup")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_1_3_1_2_1_1",
    allow_duplicates = True,
    disjoint_from = None,
)

ga321test_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga321test")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_1_3_1_2_1_1",
    allow_duplicates = False,
    disjoint_from = ga321_dataset,
)

ga321duptest_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga321duptest")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_1_3_1_2_1_1",
    allow_duplicates = True,
    disjoint_from = ga321dup_dataset,
)


ga321prop12dup_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga321prop12dup")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
    sampler = "GlobalAvoidance_1_3_1_2_1_1",
    allow_duplicates = True,
    disjoint_from = None,
)

ga321prop12duptest_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga321prop12duptest")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
    sampler = "GlobalAvoidance_1_3_1_2_1_1",
    allow_duplicates = True,
    disjoint_from = ga321prop12dup_dataset,
)


a32prop12dup_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "a32prop12dup")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
    sampler = "UntilTaskSampler_1_3_1_2",
    allow_duplicates = True,
    disjoint_from = None,
)

a32prop12duptest_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "a32prop12duptest")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
    sampler = "UntilTaskSampler_1_3_1_2",
    allow_duplicates = True,
    disjoint_from = a32prop12dup_dataset,
)


datasets = [

    e54_dataset,
    e54dup_dataset,
    e54test_dataset,
    e54duptest_dataset,

    e54prop12dup_dataset,
    e54prop12duptest_dataset,

    ga321_dataset,
    ga321dup_dataset,
    ga321test_dataset,
    ga321duptest_dataset,

    ga321prop12dup_dataset,
    ga321prop12duptest_dataset,

    a32prop12dup_dataset,
    a32prop12duptest_dataset,

]


# Only samplers:

# e65hard -> Eventually_6_6_5_5
# e+depth -> Eventually_1_15_1_4
# e+conjs -> Eventually_1_5_1_12

# ga431hard -> GlobalAvoidance_4_4_3_3_1_1
# ga+depth -> GlobalAvoidance_1_6_1_2_1_1
# ga+conjs -> GlobalAvoidance_1_3_1_3_1_1


def get_dataset(path: Path) -> Dataset:
    resolved_path = path.resolve()
    for dataset in datasets:
        if dataset.path.resolve() == resolved_path:
            return dataset
    raise ValueError(f"Dataset {path} not found.")



# Warning: because of how the LTLf2DFA works, having more than one process creating automata at 
# the same time may cause inconsistencies and errors

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="e54dup")
    parser.add_argument("--target", type=str, default="both", choices=["formulas", "automata", "both"])
    args = parser.parse_args()

    dataset_path = Path(os.path.join(DATASETS_DIR, args.name))
    target = args.target
    dataset = get_dataset(dataset_path)

    if target == 'formulas':
        dataset.save_formulas()
    elif target == 'automata':
        dataset.save_automata()
    elif target == 'both':
        dataset.save_formulas()
        dataset.save_automata()
    else:
        print(f"Unknown target {target}, expected 'formulas', 'automata' or 'both'.")