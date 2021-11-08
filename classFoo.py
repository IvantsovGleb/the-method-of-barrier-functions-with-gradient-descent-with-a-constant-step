class Foo:
    def __init__(self, func, grad):
        self.__func = func
        self.__gradient = grad

    def get_func(self):
        return self.__func

    def get_gradient(self):
        return self.__gradient

    def check_point(self, x):
        return self.__func(x) <= 79
