import numpy as np


class MathematicalObject:
    """
    Encapsulates method overloads specific to mathematical objects and simplifies implementations of classes that derive
    from this class.
    The methods that need to be implemented in the children class are :
    __add__         __sub__         __mul__         __truediv__
    __pow__         log             exp             __abs__
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
    
    def __ipow__(self, other):
        self = self.__pow__(other)
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            if ufunc is np.log:
                return self.log()
            if ufunc is np.exp:
                return self.exp()
            if ufunc is np.abs:
                return self.__abs__()
            else:
                raise NotImplementedError(f"the ufunc {ufunc} is not implemented.")
                # return ufunc(*inputs, **kwargs)
        return NotImplemented
