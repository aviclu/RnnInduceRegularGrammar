from utils import *


def test():
    states = [SearchNode(State([i])) for i in range(7)]
    s0, s1, s2, s3, s4, s5, s6 = states
    s0.transitions = {1: s2, 2: s1}
    s1.transitions = {2: s6}
    s2.transitions = {2: s3}
    s3.transitions = {1: s4}
    s4.transitions = {2: s5}
    s5.state.final = True
    states_dict = {s0: s0.transitions, s1: s1.transitions, s2: s2.transitions, s3: s3.transitions,
                   s4: s4.transitions, s5: s5.transitions, s6: s6.transitions}
    reversed_graph_nodes = get_reverse_graph(copy(states_dict))
    reachable_nodes = get_reachable_nodes(reversed_graph_nodes)
    trimmed_graph = set.intersection(set(states_dict.keys()), reachable_nodes)
    print(len(trimmed_graph))


if __name__ == '__main__':
    test()
