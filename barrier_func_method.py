import numpy as np
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


# if (r ^ k) x  * B(x, r ^ k) < eps
# x ^ *(r ^ k) is a minimum point
def is_time_to_stop(point, rk, barrier_functions, e) -> bool:
    b = -rk * sum(map(lambda g: 1. / g.get_func(point), barrier_functions))
    print(b)
    return True if np.abs(b) < e else False


def barrier_method(f: Foo, barrier_functions, x0, r0, c) -> float:
    k = 0
    rk = r0
    xk = 0
    cur_point = x0
    eps = 0.00000000001
    condition = True
    while condition:
        auxiliary_func = get_auxiliary_func(f, barrier_functions, rk)
        xk = descent_with_const_step(auxiliary_func, cur_point)
        if all([bf.check_point(xk) for bf in barrier_functions]):
            print(
                'Current point: {},     f(xk) <= 0: {},     gradient: {}'.format(
                    xk,
                    all([bf.check_point(xk) for bf in barrier_functions]),
                    str(auxiliary_func.get_gradient(xk))
                )
            )
        else:
            print('diverge', xk)
        condition = not is_time_to_stop(xk, rk, barrier_functions, eps)

        if condition:
            rk = rk / c  # r ^ k, k =  k + 1
            cur_point = xk
            k += 1
    return xk
