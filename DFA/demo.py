import DFA as dfa
import numpy as np
from search_node import SearchNode

# 5 / 10 8 7 3

# Basics:
from state import State

states = range(5)
start = 0
accepts = [0]
alphabet = ['0', '1']


def delta(state, char):
    char = int(char)
    if char == 0:
        return state * 2 % 5
    return (state * 2 + 1) % 5


d = dfa.DFA(states=states, start=start, accepts=accepts, alphabet=alphabet, delta=delta)
print("Given a binary input, d accepts if the number represented is divisible by 5 (plus the empty string):")
d.pretty_print()
# raw_input()
print('d.input_sequence("1110101011101") #7517')
d.input_sequence("1110101011101")  # 7517
print("Current state:", d.current_state)
print("Accepting:", d.status())
# raw_input()
print("Resetting...")
d.reset()
print(d.current_state)
d.input_sequence("10011011101")  # 1245
print(d.current_state)
print(d.status())

# Various minimizations
a = ['1', '11', '111', '1111', '11110', '11111', '111111', '111110']
b = ['0', '1']
e = dfa.from_word_list(a, b)
print("a = ['1', '11', '111', '1111', '11110', '11111', '111111', '111110']")
print("b = ['0', '1']")
print("e = DFA.from_word_list(a,b)")
print("...")
# raw_input()
print("===The starting DFA===")
e.pretty_print()
# raw_input()
print("==Minimized===")
e.minimize()
e.pretty_print()
# raw_input()
print("==...then DFCA-Minimized===")
e.DFCA_minimize()
e.pretty_print()
# raw_input()
print("==...then Finite-Difference Minimized===")
e.hyper_minimize()
e.pretty_print()
# raw_input()

# States DFA:
s1 = SearchNode(State(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
s2 = SearchNode(State(np.array([1., 1., 1.]), np.array([1., 1., 1.])))
s3 = SearchNode(State(np.array([1., 1., 0.]), np.array([1., 1., 0.])))
states = [s1, s2, s3]
start = SearchNode(State(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
accepts = [s2]
alphabet = ['0', '1']


def delta(state, char):
    char = int(char)
    if char == 0:
        return SearchNode(State(np.array([1., 1., 1.]), np.array([1., 1., 1.])))
    return SearchNode(State(np.array([1., 1., 0.]), np.array([1., 1., 0.])))

d = dfa.DFA(states=states, start=start, accepts=accepts, alphabet=alphabet, delta=delta)
print("the DFA accepts all binary strings that end with 0")
d.pretty_print()
# raw_input()
print('d.input_sequence("1110101011101") #7517')
d.input_sequence("1110101011101")  # 7517
print("Current state:", d.current_state)
print("Accepting:", d.status())  # should be false
# raw_input()
print("Resetting...")
d.reset()
print(d.current_state)
d.input_sequence("1001101110")  # 1245
print(d.current_state)  # should be [1,1,1]
print(d.status())
d.minimize()
d.pretty_print()  # there should be only 2 states now

