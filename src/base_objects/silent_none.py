from src.base_objects.mathematical_object import MathematicalObject


class SilentNone(MathematicalObject):
    """
    This class implements the SilentNone object, which is the equivalent of the None object. This used as a placeholder
    for Maps or Cubes without uncertainties or headers. Calling methods, operations, getitems, etc. on this object will
    never raise an error.
    """
    def __getattr__(self, _):
        return self
    
    def __getattribute__(self, _):
        return self
    
    def __call__(self, *args, **kwargs):
        return self
    
    def __getitem__(self, _):
        return self
    
    def __setitem__(self, _, value):
        return self
    
    def __repr__(self):
        return "SilentNone()"
    
    def __str__(self):
        return "SilentNone"

    def __bool__(self):
        return False
    
    def __eq__(self, other):
        return other is None or isinstance(other, SilentNone)

    def __add__(self, _):
        return self

    def __sub__(self, _):
        return self

    def __mul__(self, _):
        return self

    def __truediv__(self, _):
        return self

    def __pow__(self, _):
        return self

    def __abs__(self, _):
        return self

    def log(self):
        return self

    def exp(self):
        return self
