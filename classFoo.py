from matplotlib import pyplot as plt
import numpy as np


class Foo(object):
    def __init__(self, func, gradient=None):
        self.__func = func
        self.__gradient = gradient

    @property
    def get_func(self):
        return self.__func

    @property
    def get_gradient(self):
        return self.__gradient

    def check_point(self, x):
        return self.__func(x) <= 0

    def draw_func(self):
        l, r = -10, 10
        inputs = np.arange(l, r, 0.1)
        results = self.get_func(inputs)
        plt.plot(inputs, results)
