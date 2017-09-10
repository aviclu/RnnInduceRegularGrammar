import numpy as np
from config import Config

config = Config()


class State(object):
    def __init__(self, vec, quantized=None):
        self.__vec = vec
        self.__quantized = quantized
        self.__is_final = False
        self.__is_legal = True

    @property
    def vec(self):
        return self.__vec

    @property
    def quantized(self):
        if self.__quantized is None:
            self.quantize()
        return tuple(self.__quantized)

    @quantized.setter
    def quantized(self, value):
        self.__quantized = value

    def quantize(self, div=4):
        self.__quantized = State.quantize_vec(self.__vec, div)

    @staticmethod
    def quantize_vec(vec, div):
        return [State.round_by_div(n, div) for n in vec]

    @staticmethod
    def round_by_div(number, div):
        return round(number * div) / div

    @property
    def legal(self):
        return self.__is_legal

    @legal.setter
    def legal(self, value):
        self.__is_legal = value

    @property
    def final(self):
        return self.__is_final

    @final.setter
    def final(self, value):
        self.__is_final = value

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.quantized == other.quantized

    def __hash__(self):
        return hash(self.quantized)

    def __lt__(self, other):
        return isinstance(other, self.__class__) and (np.array(self.vec) > np.array(other.vec)).all()

    def __repr__(self):
        return ("[" + ', '.join(['%.2f'] * len(self.quantized)) + "]") % self.quantized
