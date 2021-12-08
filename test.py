from barrier_func_method import barrier_method
from main import f, g

# from scipy.optimize import minimize
# import numpy as np
# import pulp as plp

from decimal import *

getcontext().prec = 4


def f_range(start: Decimal, stop: Decimal, step: Decimal):
    while start <= stop:
        yield Decimal(start)
        start += step


def test():
    x0 = [Decimal(0.1), Decimal(0.3), Decimal(0.3), Decimal(0.3), Decimal(0.3)]
    r0 = Decimal(10)
    c = Decimal(0.5)

    for i, xi in enumerate(x0):
        print(f'\n№ {i + 1} variable is fixed')
        print(100 * '*' + '\n')
        for j in f_range(Decimal(-10), Decimal(10), Decimal(0.1)):
            x0[i] = j
            if g.check_point(x0):
                xk = barrier_method(f, g, x0, r0, c)
                if xk != [0, 0, 0, 0, 0]:
                    print('start_point: [{x1:.1f}, {x2:.1f}, {x3:.1f}, {x4:.1f}, {x5:.1f}]'.format(x1=x0[0], x2=x0[1], x3=x0[2], x4=x0[3], x5=x0[4]))
                    print('g(x0): {} <= 79'.format(g.get_func()(x0)))
                    print('optima: [{x1:.1f}, {x2:.1f}, {x3:.1f}, {x4:.1f}, {x5:.1f}]'.format(x1=xk[0], x2=xk[1], x3=xk[2], x4=xk[3], x5=xk[4]))
                    print('g(xk): {} <= 79\n'.format(g.get_func()(xk)))
                else:
                    pass
                    # print('method diverges\n')
        print(100 * '*' + '\n')
        x0[i] = xi


def main():
    test()
    return 0


if __name__ == '__main__':
    main()
