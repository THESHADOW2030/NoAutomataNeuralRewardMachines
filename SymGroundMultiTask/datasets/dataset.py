import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from dataclasses import dataclass

from utils import ltl_ast2dfa, set_seed
from ltl_samplers import getLTLSampler


def save_verbose(obj, path, name):
    with open(path, 'wb') as f:
        print(f'Saving {name} to {path}...', end='')
        pickle.dump(obj, f)
        print('Done.')


def check_existing(path, ask_abort=True):
    if os.path.isfile(path):
        if ask_abort and input(f'File {path} already exists. Overwrite? [y/N] ').lower() != 'y':
            print('Aborting.')
            sys.exit(0)
        return True
    return False


@dataclass
class Dataset:

    path: Path
    seed: int
    n_formulas: int
    propositions: List[str]
    sampler: str
    allow_duplicates: bool
    disjoint_from: Optional['Dataset']


    def mkdir(self):
        self.path.mkdir(parents=True, exist_ok=True)


    @property
    def formulas_path(self):
        return self.path / 'formulas.pkl'


    @property
    def automata_path(self):
        return self.path / 'automata.pkl'


    @property
    def config_path(self):
        return self.path / 'config.pkl'


    def load_formulas(self):
        with open(self.formulas_path, 'rb') as f:
            formulas = pickle.load(f)
        return formulas


    def save_formulas(self, pbar=True):
        self.mkdir()
        check_existing(self.formulas_path)

        print(f'Generating {self.n_formulas} formulas using {self.sampler} sampler and propositions {self.propositions}')
        if self.disjoint_from is not None:
            print(f'Ensuring that the generated formulas are disjoint from the formulas in {self.disjoint_from.path}...')
            disjoint_formulas = self.disjoint_from.load_formulas()
        else:
            disjoint_formulas = ()

        sampler = getLTLSampler(self.sampler, self.propositions)
        set_seed(self.seed)
        formulas = []
        for _ in tqdm(range(self.n_formulas)) if pbar else range(self.n_formulas):
            formula = None
            # Ensure that the generated formulas are unique
            while formula is None:
                formula = sampler.sample()
                if (not self.allow_duplicates and formula in formulas) or (formula in disjoint_formulas):
                    formula = None
            formulas.append(formula)
        save_verbose(formulas, self.formulas_path, 'formulas')
        self.save_config()


    def save_config(self):
        self.mkdir()
        check_existing(self.config_path)
        config = {
            "path": str(self.path),
            "seed": self.seed,
            "n_formulas": self.n_formulas,
            "propositions": self.propositions,
            "sampler": self.sampler,
            "disjoint_from": self.disjoint_from
        }
        save_verbose(config, self.config_path, 'config')


    def load_automata(self):
        with open(self.automata_path, 'rb') as f:
            automata = pickle.load(f)
        return automata


    def save_automata(self, pbar=True):
        formulas = self.load_formulas()
        self.mkdir()
        check_existing(self.automata_path)

        print('Computing the DFA of each formula...')
        automata = []
        for formula in tqdm(formulas) if pbar else formulas:
            automata += [ltl_ast2dfa(formula, symbols=self.propositions)]
        save_verbose(automata, self.automata_path, 'automata')