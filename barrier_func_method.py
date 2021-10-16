import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from gradient_descent_with_const_step import descent_with_const_step
from classfoo import Foo


# F(x, r ^ k) = f(x) + B(x, r ^ k) - this is auxiliary, which we gonna minimize
# B(x, r ^ k) = -(r ^ k) * sum(i = 1,m)(1 / gi(x)) - barrier function itself
def get_auxiliary_func(f: Foo, barrier_functions, rk) -> Foo:
    return Foo(
        lambda x: f.get_func(x) - rk * sum(
            map(
                lambda g: 1. / g.get_func(x),
                barrier_functions
            )
        ),
        lambda x: f.get_gradient(x) - rk * sum(
            map(
                lambda g: -g.get_gradient(x) / (g.get_func(x) ** 2),
                barrier_functions
            )
        )
    )


# if (r ^ k) * B(x, r ^ k) < eps
# x ^ *(r ^ k) is a minimum point
def is_time_to_stop(point, rk, barrier_functions, e):
    b = sum(map(lambda g: 1. / g.get_func(point), barrier_functions))
    return tuple((True if np.abs(-rk * b) < e else False, np.abs(-rk * b)))


def barrier_method(f: Foo, barrier_functions, x0, r0, c) -> float:
    k = 0
    rk = r0
    xk = 0
    cur_point = x0
    eps = 0.00000000001
    condition = True

    table = []
    headers = ['k', 'r ^ k', 'x ^ k', 'f(xk)', 'anti-gradient f(xk)', 'F(x, r ^ k)', '(r ^ k) * B(x, r ^ k)']

    inputs = []
    results = []
    f.draw_func()
    barrier_functions.__getitem__(0).draw_func()
    while condition:
        inputs.append(xk)
        results.append(f.get_func(xk))
        plt.scatter(inputs, results)
        plt.pause(0.05)

        auxiliary_func = get_auxiliary_func(f, barrier_functions, rk)
        xk = descent_with_const_step(auxiliary_func, cur_point)
        v = is_time_to_stop(xk, rk, barrier_functions, eps)

        table.append([k, rk, xk, f.get_func(xk), -f.get_gradient(xk), auxiliary_func.get_func(xk), v[1]])

        condition = not v[0]
        if condition:
            rk = rk / c  # r ^ k, k =  k + 1
            cur_point = xk
            k += 1

    print(tabulate(table, headers, tablefmt="pretty"))
    optima_x = xk
    optima_y = f.get_func(optima_x)
    plt.plot([optima_x], [optima_y], 's', color='r')
    plt.show()

    return optima_x
