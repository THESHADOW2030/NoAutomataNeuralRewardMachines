formulas = []

items = ['pickaxe', 'lava', 'door', 'gem', 'empty' ]

#PATTERNS INSPIRED FROM LTL2action
#formulas.append(("(F c0) & (F c1)", 2, "task1: visit({0}, {1})".format(*items)))
#formulas.append(("(F c0) & (F c1) & (F c2)", 3, "task2: visit({0}, {1}, {2})".format(*items)))
#formulas.append(("F(c0 & F(c1))", 2, "task3: seq_visit({0}, {1})".format(*items)))
#formulas.append(("F(c0 & F(c1)) & (F c2)", 3, "task5: seq_visit({0}, {1}) + visit({2})".format(*items)))
#formulas.append(("(F c0) & (F c1) & (G (! c2))", 3, "task7: visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2)) & (G(! c3))", 4, "task8: visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2)", 3, "task9: seq_visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2) & G(! c3)", 4, "task10: seq_visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))


#prima seconda terza quinta settima ottava nona decima

#cleanup experiments

"""
Task 1 -> Task 1
Task 2 -> Task 2
Task 3 -> Task 3
Task 4 -> Task 5
Task 5 -> Task 7
Task 6 -> Task 8
Task 7 -> Task 9
Task 8 -> Task 10

"""