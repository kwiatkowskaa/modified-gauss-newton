import numpy as np
from abc import ABC, abstractmethod

class Problem(ABC):
    def __init__(self, name, n, m, difficulty, classification_model, 
                 source=None, certified_solution=None, certified_rss=None):
        self.name = name
        self.n = n
        self.m = m

        self.difficulty = difficulty
        self.classification_model = classification_model
        self.source = source

        self.certified_solution = certified_solution
        self.certified_rss = certified_rss

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