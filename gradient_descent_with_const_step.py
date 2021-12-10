from classFoo import Foo


def sub(xk, alpha, grad):
    return [xi - alpha * grad[i](xk) for i, xi in enumerate(xk)]


def descent_with_const_step(f: Foo, xk):
    alpha = 0.5
    return sub(xk, alpha, f.get_gradient())