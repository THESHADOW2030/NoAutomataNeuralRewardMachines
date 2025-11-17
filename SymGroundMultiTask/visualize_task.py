import os
import argparse, ast

import utils
from ltl_samplers import getLTLSampler


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="sampler", choices=["sampler", "manual"])
parser.add_argument("--sampler", default="Dataset_e54test_no-shuffle", type=str)
parser.add_argument("--id", default=0, type=int)
parser.add_argument("--formula", default=('eventually', 'b'), type=ast.literal_eval)
parser.add_argument("--symbols", nargs="+", default=["a", "b", "c", "d", "e"], type=str)
parser.add_argument('--automaton', dest='automaton', default=True, action='store_true')
parser.add_argument('--no-automaton', dest='automaton', action='store_false')
args = parser.parse_args()

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_DIR, "saves")


if args.mode == "sampler":
    sampler = getLTLSampler(args.sampler, args.symbols)
    if "Dataset" in args.sampler:
        formula = sampler.get_formula(args.id)
    else:
        formula = sampler.sample()

elif args.mode == "manual":
    formula = tuple(args.formula)


print("Formula:")
utils.pprint_ltl_formula(formula)


if args.automaton:

    if args.mode == "sampler":
        automaton = utils.ltl_ast2dfa(formula, args.symbols)

    if args.mode == "manual":
        automaton = utils.ltl_ast2dfa(formula, args.symbols)

    automaton.write_dot_file(os.path.join(OUTPUT_DIR, "automaton.dot"))
    automaton.show(os.path.join(OUTPUT_DIR, "automaton"))