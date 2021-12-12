class Foo:
    def __init__(self, func=None, grad=None):
        self.__func = func
        self.__gradient = grad

    def get_func(self):
        return self.__func

    def get_gradient(self):
        return self.__gradient

    def check_point(self, x) -> bool:
        return self.__func(x) <= 79

    def check_point_strict(self, x) -> bool:
        return self.__func(x) < 79
