from classFoo import Foo
from decimal import *

getcontext().prec = 4


def sub(xk, alpha, grad):
    return [xi - alpha * grad[i](xk) for i, xi in enumerate(xk)]


def descent_with_const_step(f: Foo, xk):
    alpha = Decimal(0.002)

    return sub(xk, alpha, f.get_gradient())
