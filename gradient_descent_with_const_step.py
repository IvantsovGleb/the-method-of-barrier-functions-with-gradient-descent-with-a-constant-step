from classFoo import Foo
from classGradient import Gradient


def sub(vector, alpha, grad: Gradient):
    return [el - alpha * grad[i](grad.get__x()) for i, el in enumerate(vector)]


def descent_with_const_step(f: Foo, xk):
    f.get_gradient().set__x(xk)
    alpha = 0.09
    return sub(xk, alpha, f.get_gradient())
