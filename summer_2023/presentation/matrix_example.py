import numpy as np


class Matrix:
    def __init__(self, data: np.ndarray):
        self.data = data

    def __add__(self, other):
        return Matrix(self.data + other.data)
    
    def __sub__(self, other):
        return Matrix(self.data - other.data)
    
    def __mul__(self, other):
        return Matrix(np.matmul(self.data, other.data))
    
    def __str__(self):
        return str(self.data)
    
a = Matrix(np.array([
    [2,3],
    [4,5]
]))
    
b = Matrix(np.array([
    [6,7],
    [8,9]
]))

print(a * b)

class Column_vector(Matrix):
    def __init__(self, data: np.ndarray):
        assert data.ndim == 1, "Data is not one-dimensional"
        self.data = data
    
    def get_norm(self):
        return np.sqrt(np.sum(self.data**2))
    
c = Column_vector(np.array([
    1,1,1
]))

print(c + c)

print(c.get_norm())
