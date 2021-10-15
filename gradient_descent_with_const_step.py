from classfoo import Foo


def descent_with_const_step(f: Foo, xk) -> float:
    #  x(k + 1) = xk - alpha * df(xk)/dx, alpha is constant
    alpha = 0.09
    return xk - alpha * f.get_gradient(xk)
