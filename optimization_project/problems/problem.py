import numpy as np
from abc import ABC, abstractmethod

class Problem(ABC):
    def __init__(self, name, n, m):
        self.name = name
        self.n = n
        self.m = m

    @abstractmethod
    def F(self, x):
        """Calculates the residual vector F(b) = model(x, b) - y. """
        pass
    
    @abstractmethod
    def J(self, x):
        """ Calculates the Jacobian matrix J(b) = dF/db. """
        pass

    @abstractmethod
    def get_starting_points(self):
        """Returns the proposed starting points defined by NIST."""
        pass