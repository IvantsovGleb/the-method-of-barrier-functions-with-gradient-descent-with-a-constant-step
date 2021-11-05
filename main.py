from classFoo import Foo
from classGradient import Gradient
from barrier_func_method import barrier_method

x0 = [1, 1, 1, 1, 1]

gradient = [
    lambda x: 802 * x[0] - 8 * (25 * (x[1] + x[2] + x[3] + x[4]) + 1),
    lambda x: 100 * (2 * x[1] - 2 * x[0]),
    lambda x: 100 * (2 * x[2] - 2 * x[0]),
    lambda x: 100 * (2 * x[3] - 2 * x[0]),
    lambda x: 100 * (2 * x[4] - 2 * x[0])
]

f = Foo(
    lambda x: 100 * ((x[1] - x[0]) ** 2 + (x[2] - x[0]) ** 2 + (x[3] - x[0]) ** 2 + (x[4] - x[0]) ** 2) + (
            x[0] - 4) ** 2,
    Gradient(gradient)
)

barrier_gradient = [
    lambda x: 2 * x[0],
    lambda x: 4 * x[1],
    lambda x: 6 * x[2],
    lambda x: 8 * x[3],
    lambda x: 10 * x[4]
]

g = Foo(
    lambda x: sum(i * xi ** 2 for i, xi in enumerate(x, start=1)),
    Gradient(barrier_gradient)
)


def main():
    barrier_functions = [g]
    r0 = 2  # [2, 10]
    c = 2

    print('optima: {}'.format(barrier_method(f, barrier_functions, x0, r0, c)))
    return 0


if __name__ == '__main__':
    main()
