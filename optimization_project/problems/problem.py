from abc import ABC, abstractmethod
import numpy as np

class Problem(ABC):
    def __init__(
        self, name, n, m, 
        difficulty=None, 
        classification_model=None, 
        source=None, 
        certified_solution=None, 
        certified_rss=None
    ):
        self.name = name
        self.n = n  # Number of variables
        self.m = m  # Number of equations/observations

        # Optional metadata (mostly used for NIST datasets)
        self.difficulty = difficulty
        self.classification_model = classification_model
        self.source = source

        # Optional target metrics (can be None for underdetermined problems)
        self.certified_solution = certified_solution
        self.certified_rss = certified_rss

    @property
    def is_underdetermined(self):
        """Automatically checks the system geometry on the fly."""
        return self.m <= self.n

    @abstractmethod
    def F(self, x):
        """Calculates the residual vector F(x) of shape (m,)."""
        pass
    
    @abstractmethod
    def J(self, x):
        """Calculates the Jacobian matrix J(x) of shape (m, n)."""
        pass

    @abstractmethod
    def get_starting_points(self):
        """Returns a dictionary of starting points."""
        pass