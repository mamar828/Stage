class MathematicalObject:
    """
    Encapsulates method overloads specific to mathematical objects and simplifies implementations of classes that derive
    from this class.
    The __add__, __sub__, __mul__ and __truediv__ methods must be implemented in the children class.
    """
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def __rsub__(self, other):
        return self.__sub__(other) * (-1)
    
    def __isub__(self, other):
        self = self.__sub__(other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        self = self.__mul__(other)
        return self
    
    def __rtruediv__(self, other):
        return self.__truediv__(other) ** (-1)
    
    def __itruediv__(self, other):
        self = self.__truediv__(other)
        return self
    