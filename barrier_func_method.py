from tabulate import tabulate

from classFoo import Foo
from gradient_descent_with_const_step import descent_with_const_step
from decimal import *

getcontext().prec = 4


def get_auxiliary_function(f: Foo, g, rk) -> Foo:
    grad = []
    for der_f, der_g in zip(f.get_gradient(), g.get_gradient()):
        grad.append(lambda x: der_f(x) - rk * -der_g(x) / g.get_func()(x) ** 2)

    return Foo(lambda x: f.get_func()(x) - rk * Decimal(1) / g.get_func()(x), grad)


def is_time_to_stop(xk, rk, g):
    eps = Decimal(0.00001)
    B = Decimal(-1) / g.get_func()(xk)
    return True if abs(rk * B) <= eps else False, abs(rk * B)


def barrier_method(f: Foo, g, x0, r0, c):
    xk = 0

    table = []
    headers = ['k', 'r ^ k', 'x ^ k', 'f(xk)', 'F(x, r ^ k)', '(r ^ k) * B(x, r ^ k)']
    k = Decimal(0)
    rk = r0
    cur_point = x0
    condition = True
    while condition:
        auxiliary_func = get_auxiliary_function(f, g, rk)
        xk = descent_with_const_step(auxiliary_func, cur_point)
        v = is_time_to_stop(xk, rk, g)

        table.append([k, rk, xk, f.get_func()(xk), auxiliary_func.get_func()(xk), v[1]])

        condition = not v[0]
        if condition:
            rk = rk * c
            cur_point = xk
            k += 1

    print(tabulate(table, headers, tablefmt="pretty"))
    return xk
