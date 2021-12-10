from classFoo import Foo
from gradient_descent_with_const_step import descent_with_const_step

from tabulate import tabulate
import numpy as np


def get_auxiliary_function(f: Foo, g, t) -> Foo:
    grad = []
    for der_f, der_g in zip(f.get_gradient(), g.get_gradient()):
        grad.append(lambda x: t * der_f(x) - der_g(x) / g.get_func()(x))

    return Foo(lambda x: t * f.get_func()(x) + get_barrier_function(g)(x), grad)


def get_barrier_function(g: Foo):
    return lambda x: -np.log(g.get_func()(x))


def barrier_method(f: Foo, g: Foo, x0, t0, gamma) -> (list, int):

    table = []
    headers = ['k', 't ^ k', 'f(xk)', 'B(xk)', 'F( r ^ k, x ^ k)']

    xk = [0, 0, 0, 0, 0]
    k = 0
    tk = t0
    cur_point = x0

    eps = 0.0000001
    condition = True
    while condition:
        auxiliary_func = get_auxiliary_function(f, g, tk)
        xk = descent_with_const_step(auxiliary_func, cur_point)

        if auxiliary_func.gradient_norm(xk) <= eps * auxiliary_func.gradient_norm(cur_point):
            condition = False

        table.append([k, tk, f.get_func()(xk), get_barrier_function(g)(xk), auxiliary_func.get_func()(xk)])

        if condition:
            tk = gamma * tk
            cur_point = xk
            k += 1

    # if g.check_point(xk):
    print(tabulate(table, headers, tablefmt="pretty"))
    print('\n')
    return xk, k
    # return [0, 0, 0, 0, 0], 0
