from classFoo import Foo
from gradient_descent_with_const_step import descent_with_const_step

import numpy as np
from tabulate import tabulate
from decimal import *

getcontext().prec = 4


# def get_auxiliary_function(f: Foo, g, rk) -> Foo:
#     grad = []
#     for der_f, der_g in zip(f.get_gradient(), g.get_gradient()):
#         grad.append(lambda x: der_f(x) + rk * der_g(x) / g.get_func()(x) ** 2)
#
#     return Foo(lambda x: f.get_func()(x) + rk * get_barrier_function(g)(x), grad)
#
#
# def get_barrier_function(g: Foo):
#     return lambda x: -Decimal(1) / g.get_func()(x)

def get_auxiliary_function(f: Foo, g, t) -> Foo:
    grad = []
    for der_f, der_g in zip(f.get_gradient(), g.get_gradient()):
        grad.append(lambda x: t * der_f(x) - der_g(x) / g.get_func()(x))

    return Foo(lambda x: t * f.get_func()(x) + Decimal(get_barrier_function(g)(x)), grad)


def get_barrier_function(g: Foo):
    return lambda x: -np.log(float(g.get_func()(x)))


def rkB(xk, rk, g):
    rkb = rk * Decimal(get_barrier_function(g)(xk))
    return abs(rkb)


def barrier_method(f: Foo, g: Foo, x0, r0, c) -> (list, int):

    table = []
    headers = ['k', 'r ^ k', 'f(xk)', 'B(xk)', '(r ^ k) * B(x, r ^ k)', 'F( r ^ k, x ^ k)']

    xk = [0, 0, 0, 0, 0]
    k = Decimal(0)
    rk = r0
    cur_point = x0

    eps = Decimal(0.00001)
    condition = True
    while condition:
        auxiliary_func = get_auxiliary_function(f, g, rk)
        xk = descent_with_const_step(auxiliary_func, cur_point)
        rkb = rkB(xk, rk, g)
        if rkb < eps:
            condition = False

        if g.check_point(xk):
            table.append([k, rk, f.get_func()(xk), get_barrier_function(g)(xk), rkb, auxiliary_func.get_func()(xk)])

        if condition:
            rk = rk * c
            cur_point = xk
            k += 1

    if g.check_point(xk):
        print(tabulate(table, headers, tablefmt="pretty"))
        return xk, k
    return [0, 0, 0, 0, 0], 0
