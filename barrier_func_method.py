import numpy as np
from tabulate import tabulate

from classFoo import Foo
from classGradient import Gradient
from gradient_descent_with_const_step import descent_with_const_step


# F(x, r ^ k) = f(x) + B(x, r ^ k) - this is auxiliary, which we gonna minimize
# B(x, r ^ k) = -(r ^ k) * sum(i = 1,m)(1 / gi(x)) - barrier function itself
def get_auxiliary_function(f: Foo, barrier_functions, rk) -> Foo:
    grad = [
        lambda x: f.get_gradient()[0](x) - rk * sum(map(lambda g: - g.get_gradient()[0](x) / g.get_func()(x) ** 2,
                                                        barrier_functions)),
        lambda x: f.get_gradient()[1](x) - rk * sum(map(lambda g: - g.get_gradient()[1](x) / g.get_func()(x) ** 2,
                                                        barrier_functions)),
        lambda x: f.get_gradient()[2](x) - rk * sum(map(lambda g: - g.get_gradient()[2](x) / g.get_func()(x) ** 2,
                                                        barrier_functions)),
        lambda x: f.get_gradient()[3](x) - rk * sum(map(lambda g: - g.get_gradient()[3](x) / g.get_func()(x) ** 2,
                                                        barrier_functions)),
        lambda x: f.get_gradient()[4](x) - rk * sum(map(lambda g: - g.get_gradient()[4](x) / g.get_func()(x) ** 2,
                                                        barrier_functions)),
    ]
    return Foo(
        lambda x: f.get_func()(x) - rk * sum(map(lambda g: 1. / g.get_func()(x), barrier_functions)),
        Gradient(grad)
    )


# if (r ^ k) * B(x, r ^ k) < eps
# x ^ k is a minimum point
def is_time_to_stop(point, rk, barrier_functions, e):
    b = sum(map(lambda g: 1. / g.get_func()(point), barrier_functions))
    return tuple((True if np.abs(-rk * b) < e else False, np.abs(-rk * b)))


def barrier_method(f: Foo, barrier_functions, x0, r0, c):
    xk = 0

    table = []
    headers = ['k', 'r ^ k', 'x ^ k', 'f(xk)', 'anti-gradient f(xk)', 'F(x, r ^ k)', '(r ^ k) * B(x, r ^ k)']

    k = 0
    rk = r0
    cur_point = x0
    eps = 0.00000000001
    condition = True
    while condition:
        auxiliary_func = get_auxiliary_function(f, barrier_functions, rk)
        xk = descent_with_const_step(auxiliary_func, cur_point)
        v = is_time_to_stop(xk, rk, barrier_functions, eps)

        table.append([k, rk, xk, f.get_func()(xk), auxiliary_func.get_func()(xk), v[1]])
        # f.get_gradient().get__v(xk),

        condition = not v[0]
        if condition:
            rk = rk / c  # r ^ k, k =  k + 1
            cur_point = xk
            k += 1

    print(tabulate(table, headers, tablefmt="pretty"))
    return xk
