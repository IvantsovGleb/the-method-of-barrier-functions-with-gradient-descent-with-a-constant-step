from classFoo import Foo
from gradient_descent_with_const_step import gradient_descent

from tabulate import tabulate
from decimal import *

getcontext().prec = 4


def logarithmic_barrier(g: Foo) -> Foo:
    return Foo(
        lambda x: -(g.get_func()(x).ln()),
        [lambda x: - (der_g(x) / g.get_func()(x)) for der_g in g.get_gradient()]
    )


def inverse_barrier(g: Foo) -> Foo:
    return Foo(
        lambda x: Decimal(-1) / g.get_func()(x),
        [lambda x: der_g(x) / (g.get_func()(x) ** 2) for der_g in g.get_gradient()]
    )


def get_barrier_function(g: Foo, barrier_type='logarithmic'):
    if 'inverse' == barrier_type:
        return inverse_barrier(g)
    return logarithmic_barrier(g)


def get_auxiliary_function(f: Foo, g: Foo, mu, barrier_type='logarithmic') -> Foo:
    obj = get_barrier_function(g, barrier_type)
    barrier = obj.get_func()
    barrier_gradient = obj.get_gradient()
    return Foo(
        lambda x: f.get_func()(x) + mu * barrier(x),
        [lambda x: der_f(x) + mu * der_b(x) for der_f, der_b in zip(f.get_gradient(), barrier_gradient)]
    )


def method_status(xk, mu, g, barrier_type='logarithmic'):
    eps = 0.00001
    muB = mu * get_barrier_function(g, barrier_type).get_func()(xk)
    return True if abs(muB) < eps else False

# def method_status(auxiliary_func, xk, xk_plus_1):
#     eps = 0.00001
#     return True if abs(auxiliary_func.get_func()(xk_plus_1) - auxiliary_func.get_func()(xk)) <= eps else False



def barrier_method(f: Foo, g: Foo, x0, mu0, beta, barrier_type='logarithmic', points='valid') -> (list, int):
    table = []
    headers = ['k', 'mu', 'f(xk)', 'B(xk)', 'F( r ^ k, x ^ k)']

    k = 0
    mu = mu0
    cur_point = x0

    condition = True
    while condition:
        auxiliary_func = get_auxiliary_function(f, g, mu, barrier_type)
        xk = gradient_descent(auxiliary_func, cur_point)

        if points == 'valid':
            if g.check_point_strict(xk):
                condition = not method_status(xk, mu, g, barrier_type)
                # condition = not method_status(auxiliary_func, cur_point, xk)

                if condition:
                    mu = mu * beta
                    cur_point = xk
                    k += 1
            else:
                return [0, 0, 0, 0, 0], 0
        elif points == 'all':
            condition = not method_status(xk, mu, g, barrier_type)
            # condition = not method_status(auxiliary_func, cur_point, xk)

        if condition:
                mu = mu * beta
                cur_point = xk
                k += 1

        table.append([k, mu, f.get_func()(xk), get_barrier_function(g, barrier_type).get_func()(xk), auxiliary_func.get_func()(xk)])

    print(tabulate(table, headers, tablefmt="pretty"))
    return cur_point, k
