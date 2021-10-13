import numpy as np
from classfoo import Foo
from barrier_func_method import barrier_method

f = Foo(lambda x: x, lambda x: 1)  # f(x)
g = Foo(lambda x: x - 3)  # gi(x) <= 0, i = 1,m


def main():
    barrier_functions = [g]
    x0 = 1  # ?
    r0 = 2  # [2, 10]
    c = 2

    # print(f.get_func(1))
    # print(f.get_gradient(1))
    print("RES")
    print(barrier_method(f, barrier_functions, x0, r0, c))
    return 0


if __name__ == '__main__':
    main()
