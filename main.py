from classfoo import Foo
from barrier_func_method import barrier_method

f = Foo(lambda x: x ** 2, lambda x: 2*x)
g = Foo(lambda x: -x ** 2 + 10, lambda x: -2 * x)  # gi(x), i = 1,m


def main():
    barrier_functions = [g]
    x0 = 10
    r0 = 2  # [2, 10]
    c = 2

    print('optima: {}'.format(barrier_method(f, barrier_functions, x0, r0, c)))
    return 0


if __name__ == '__main__':
    main()
