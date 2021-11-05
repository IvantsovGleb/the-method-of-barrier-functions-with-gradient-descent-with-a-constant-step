class Gradient(object):
    def __init__(self, vector: list, x: list = None):
        self.__vector = vector
        self.__x = x

    def __getitem__(self, item):
        return self.__vector[item]

    def get__v(self):
        return self.__vector

    def get__x(self):
        return self.__x

    def set__x(self, value):
        self.__x = value

    # def __mul__(self, other: float):
    #     pass
