import DFA.DFA as DFA
import pydot
from data_utils import get_data_alphabet
from main import init_state
from search_node import SearchNode
from state import State
from config import Config
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


config = Config()


def color(node, init_node):
    if node.state.final:
        return 'green'
    # if node == init_node:
    #     return 'blue'
    return 'red'


def print_graph(nodes_list, graph_name, init_node=None):
    _, inv_alphabet_map = get_data_alphabet()
    graph = pydot.Dot(graph_type='digraph')
    nodes_dict = dict()
    i = 1

    for node in nodes_list:
        if node == init_node:
            label = 0
        else:
            label = i
            i += 1
        nodes_dict[node] = pydot.Node(label, style="filled", fillcolor=color(node, init_node))
        graph.add_node(nodes_dict[node])

    for node in nodes_list:
        trans = node.transitions
        next_state_to_inputs = {value: [] for value in trans.values()}
        for input_char in trans:  # merge edges that lead to the same next node
            next_state = trans[input_char]
            next_state_to_inputs[next_state].append(str(inv_alphabet_map[input_char]))
        for next_state in next_state_to_inputs:
            graph.add_edge(pydot.Edge(nodes_dict[node], nodes_dict[next_state],
                                      label=', '.join(next_state_to_inputs[next_state])))
            graph.set_edge_defaults()
        # for input in trans.keys():
        #     next_state = trans[input]
        #     graph.add_edge(pydot.Edge(nodes_dict[node], nodes_dict[next_state], label=str(inv_alphabet_map[input])))
        #     graph.set_edge_defaults()
    graph.write_png(graph_name)


def minimize_dfa(nodes, start):
    """
    building a DFA automaton from the nodes, and returning an equal graph using MN algorithm.
    :param nodes: the original nodes.
    :return: the minimized graph's set of nodes.
    """
    fail_vec = init_state - 2  # a "garbage" state
    fail_state = SearchNode(State(fail_vec, quantized=tuple(fail_vec)))
    nodes.update({fail_state: {}})

    states = nodes
    # start = SearchNode(State(init_state, quantized=tuple(init_state)))
    accepts = [node for node in nodes if node.is_accept]
    alphabet = set()
    for node in nodes:  # making sure this is the alphabet that is actually being used
        alphabet.update(key for key in node.transitions)

    def delta(state, char):
        return nodes[state].get(char, fail_state)  # return fail state in case no transition is set

    d = DFA.DFA(states=states, start=start, accepts=accepts, alphabet=alphabet, delta=delta)
    eq_classes = d.minimize()
    for state_class in eq_classes:  # updating the nodes MN representatives
        representative = state_class[0]
        for node in state_class:
            node.representative = representative

    # update the node's transitions by the new delta function of the minimized DFA
    for node in d.states:
        new_transitions = {}
        for char in alphabet:
            new_transitions[char] = d.delta(node, char)
        node.transitions = new_transitions
    return d.states


def get_reverse_graph(nodes_dict):
    """
    returns the reverse graph of the nodes dict.
    for example, for the entry s1: {1, s2} - the output dict will contain: s2: {1, s1}
    :param nodes_dict: a dictionary that maps nodes to their transitions.
    :return: the reversed graph's dictionary.
    """
    flipped_nodes = {node: {} for node in nodes_dict}
    for state, trans in nodes_dict.items():
        for char, next_state in trans.items():
            if char in flipped_nodes[next_state]:  # in case two edges with the same char are directed to this node
                char += 10
            flipped_nodes[next_state][char] = state

    output_nodes = set()
    for node in flipped_nodes.keys():  # updating the nodes transitions
        node.transitions = flipped_nodes[node]
        output_nodes.add(node)
    return output_nodes


def bfs(start_node):
    visited, queue = set(), [start_node]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vertex_neighbors = set(vertex.transitions[key] for key in vertex.transitions)
            queue.extend(vertex_neighbors - visited)
    return visited


def get_reachable_nodes(reversed_graph_nodes):
    """
    returns nodes that can be accessed by the accepting states.
    :param reversed_graph_nodes: the nodes of the reversed graph.
    :return: a set of reachable nodes.
    """
    accepting_nodes = [node for node in reversed_graph_nodes if node.is_accept]
    reachable = set()
    for node in accepting_nodes:
        reachable = set.union(reachable, bfs(node))
    return reachable


def copy(nodes_dict):
    """
    returns a dictionary that includes a copy of the nodes in the original nodes_dict.
    :param nodes_dict: a dictionary that maps every node to its transitions.
    :return: a copy of the original dict, such that it contains the same entries (with the new nodes)
    """
    output = {}
    old_to_new_states = {node: SearchNode(node.state) for node in nodes_dict}
    for state, trans in nodes_dict.items():
        if state not in output:
            output[old_to_new_states[state]] = {}
        for char, next_state in trans.items():
            output[old_to_new_states[state]][char] = old_to_new_states[next_state]
    return output


def get_trimmed_graph(analog_nodes):
    """
    trim the original graph - trim nodes that do not lead to any accepting node (not reachable).
    we do this by running bfs from the accepting nodes in the reversed graph.
    :param analog_nodes: all the nodes
    :return: all the nodes in the trimmed graph
    """
    graph_copy = copy(analog_nodes)
    assert len(graph_copy) == len(analog_nodes)
    reversed_graph_nodes = get_reverse_graph(graph_copy)
    reachable_nodes = get_reachable_nodes(reversed_graph_nodes)  # get all the states that lead to accepting states
    trimmed_graph = set()
    for node in analog_nodes:
        if node in reachable_nodes:
            trimmed_graph.add(node)
            new_trans = {char: next_node for char, next_node in node.transitions.items()
                         if next_node in reachable_nodes}
            node.transitions = new_trans  # updating the node's transitions so it will not lead to dead nodes
    assert len([node for node in trimmed_graph if node.is_accept]) == len(
        [node for node in analog_nodes if node.is_accept])
    return trimmed_graph


def evaluate_graph(X, y, init_node):
    acc = 0
    errors = []
    for sent, label in zip(X, y):
        curr_node = init_node
        for word in sent:
            curr_node = curr_node.transitions[word]
        prediction = curr_node.is_accept
        acc += (prediction == label)
        if prediction != label:
            errors.append(sent)
    return acc / len(y), errors


def is_accurate(X, y, init_node):
    for sent, label in zip(X, y):
        curr_node = init_node
        for word in sent:
            curr_node = curr_node.transitions[word]
        prediction = curr_node.is_accept
        if prediction != label:
            return False
    return True


def plot_states(states, colors, title, pca_model, path, is_quantized_graph=False):
    transformmed_X = pca_model.transform(states)
    plt.scatter(transformmed_X[:, 0], transformmed_X[:, 1], c=colors)
    plt.title(title)
    if is_quantized_graph:
        accept_legend = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
        reject_legend = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red")
        # trimmed_legend = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue")
        plt.legend((accept_legend, reject_legend), ('accept state', 'reject state'), numpoints=1, loc=1)

    plt.savefig(path + title + '.png')
