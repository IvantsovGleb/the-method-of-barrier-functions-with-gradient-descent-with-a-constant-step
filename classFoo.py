class Foo:
    def __init__(self, func, grad):
        self.__func = func
        self.__gradient = grad

    def get_func(self):
        return self.__func

    def get_gradient(self):
        return self.__gradient

    def gradient_norm(self, xk):
        sum_of_squares = 0
        for der in self.get_gradient():
            sum_of_squares += der(xk) * der(xk)
        return sum_of_squares ** 0.5

    def check_point(self, x) -> bool:
        return self.__func(x) <= 79

    def check_point_strict(self, x) -> bool:
        return self.__func(x) < 79
