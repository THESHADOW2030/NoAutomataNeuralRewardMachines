from SymGroundMultiTask.finite_state_machine import MooreMachine
from pyparsing import Word, alphas, infixNotation, opAssoc, ParserElement, alphanums
import re

ParserElement.enablePackrat()


# input LTL syntax is from LTL2ACtion (https://github.com/LTL2Action/LTL2Action)
# output LTL syntax is from LTLF2DFA (http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax)
# with the addition of parenthesis for grouping
def ltl_ast2str(ast) -> str:
    if not isinstance(ast, tuple):
        assert isinstance(ast, str)
        return ast
    op, *args = ast
    if op == 'or':
        return f"({ltl_ast2str(args[0])}) | ({ltl_ast2str(args[1])})"
    elif op == 'until':
        return f"({ltl_ast2str(args[0])}) U ({ltl_ast2str(args[1])})"
    elif op == 'and':
        return f"({ltl_ast2str(args[0])}) & ({ltl_ast2str(args[1])})"
    elif op == 'not':
        return f"!({ltl_ast2str(args[0])})"
    elif op == 'eventually':
        return f"F ({ltl_ast2str(args[0])})"
    elif op == 'always':
        return f"G ({ltl_ast2str(args[0])})"
    elif op == 'next':
        return f"X ({ltl_ast2str(args[0])})"


# from LTL2Action formula to MooreMachine
def ltl_ast2dfa(ltl_ast, symbols, name='placeholder'):
    ltl = ltl_ast2str(ltl_ast)
    return MooreMachine(
        ltl,
        len(symbols),
        name,
        dictionary_symbols=symbols,
        reward='ternary'
    )


# input LTL syntax is from LTL2ACtion (https://github.com/LTL2Action/LTL2Action)
# output LTL syntax is from LTLF2DFA (http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax)
def ltl_str2ast(ltl_str: str):

    var = Word(alphas, alphanums)

    def parse_not(tokens):
        return ('not', tokens[0][1])
    
    def parse_and(tokens):
        return ('and', tokens[0][0], tokens[0][2])

    def parse_or(tokens):
        return ('or', tokens[0][0], tokens[0][2])
    
    def parse_until(tokens):
        return ('until', tokens[0][0], tokens[0][2])
    
    def parse_eventually(tokens):
        return ('eventually', tokens[0][1])

    def parse_always(tokens):
        return ('always', tokens[0][1])

    def parse_next(tokens):
        return ('next', tokens[0][1])

    expr = infixNotation(var, [
        ('!', 1, opAssoc.RIGHT, parse_not),
        ('F', 1, opAssoc.RIGHT, parse_eventually),
        ('G', 1, opAssoc.RIGHT, parse_always),
        ('X', 1, opAssoc.RIGHT, parse_next),
        ('&', 2, opAssoc.LEFT, parse_and),
        ('|', 2, opAssoc.LEFT, parse_or),
        ('U', 2, opAssoc.LEFT, parse_until),
    ])

    parsed = expr.parseString(ltl_str, parseAll=True)
    return parsed[0]



def test():
    from ltl_samplers import DefaultSampler
    from pprint import pprint
    for i in range(5):
        print(f"formula {i}")
        ast = DefaultSampler(['a', 'b', 'c', 'd', 'e', 'f']).sample()
        print('ast:')
        pprint(ast)
        print('ltl:')
        string = ltl_ast2str(ast)
        print(string)
        print('new ast:')
        new_ast = ltl_str2ast(string)
        pprint(new_ast)
        print("\n")


if __name__ == '__main__':
    test()