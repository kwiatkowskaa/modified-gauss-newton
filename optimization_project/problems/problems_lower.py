from .problem import Problem
import numpy as np

class DanWoodProblem(Problem):
    """
    Source: NIST
    Level of Difficulty: Lower
    Model Classification: Miscellaneous
    Number of Parameters: 2
    Number of Observations: 6
    """
    def __init__(self):
        super().__init__(name="DanWood", n=2, m=6)

        self.x_data = np.array([1.309, 1.471, 1.490, 1.565, 1.611, 1.680])
        self.y_data = np.array([2.138, 3.421, 3.597, 4.340, 4.882, 5.660])

    def model(self, b, x):
        return b[0] * x**b[1]

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        J[:,0] = x**b[1]
        J[:,1] = b[0] * x**b[1] * np.log(x)

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([1.0, 5.0]),
            "NIST Start 2": np.array([0.7, 4.0])
        }