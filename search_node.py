from config import Config
from clustering_handler import get_cluster
from state import State

config = Config()


class SearchNode(object):
    def __init__(self, node_state):
        self.__state = node_state
        self.__transitions = dict()
        self.__representative = None  # MN representative

    @property
    def state(self):
        return self.__state

    @property
    def transitions(self):
        return self.__transitions

    @transitions.setter
    def transitions(self, value):
        self.__transitions = value

    @property
    def is_accept(self):
        return self.__state.final

    @property
    def representative(self):
        return self.__representative

    @representative.setter
    def representative(self, value):
        self.__representative = value

    def get_next_nodes(self, net, alphabet, old_to_new_nodes, model=None):
        next_nodes = set()
        for char in alphabet:
            next_state_analog = net.get_next_state(list(self.__state.quantized), char)
            next_state_quantized = get_cluster(next_state_analog, model) \
                if model else \
                State.quantize_vec(next_state_analog, config.States.intervals_num.int)
            # the next state is in its quantized form, so the transitions will be deterministically set -
            # hypothetically, the net may return two different actions for two different states of one cluster.
            next_state = State(next_state_quantized, next_state_quantized)

            next_node = SearchNode(next_state)
            if next_node not in old_to_new_nodes:
                old_to_new_nodes[next_node] = next_node
            next_node = old_to_new_nodes[next_node]
            self.__transitions[char] = next_node
            next_nodes.add(next_node)
        return next_nodes

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__state == other.state

    def __hash__(self):
        return hash(self.__state)

    def __lt__(self, other):
        return isinstance(other, self.__class__) and self.state < other.state

    def __repr__(self):
        return str(self.__state) + "\n" + "\n".join([str(trans) for trans in self.__transitions]) + "\n\n"
