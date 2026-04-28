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


class MGH17Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Average
    Model Classification: Exponential
    Number of Parameters: 5
    Number of Observations: 33
    """
    def __init__(self):
        super().__init__(name="MGH17", n=5, m=33)

        self.x_data = np.array([
            0,10,20,30,40,50,60,70,80,90,
            100,110,120,130,140,150,160,170,180,190,
            200,210,220,230,240,250,260,270,280,290,
            300,310,320
        ], dtype=float)

        self.y_data = np.array([
            0.844,0.908,0.932,0.936,0.925,0.908,0.881,0.850,0.818,0.784,
            0.751,0.718,0.685,0.658,0.628,0.603,0.580,0.558,0.538,0.522,
            0.506,0.490,0.478,0.467,0.457,0.448,0.438,0.431,0.424,0.420,
            0.414,0.411,0.406
        ])

    def model(self, b, x):
        return (
            b[0]
            + b[1] * np.exp(-x * b[3])
            + b[2] * np.exp(-x * b[4])
        )

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        exp4 = np.exp(-x * b[3])
        exp5 = np.exp(-x * b[4])

        J[:, 0] = 1.0
        J[:, 1] = exp4
        J[:, 2] = exp5

        J[:, 3] = -b[1] * x * exp4
        J[:, 4] = -b[2] * x * exp5

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([50.0, 150.0, -100.0, 1.0, 2.0]),
            "NIST Start 2": np.array([0.5, 1.5, -1.0, 0.01, 0.02])
        }


class Roszman1Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Average
    Model Classification: Miscellaneous
    Number of Parameters: 4
    Number of Observations: 25
    """
    def __init__(self):
        super().__init__(name="Roszman1", n=4, m=25)

        self.x_data = np.array([
            -4868.68, -4868.09, -4867.41,
            -3375.19, -3373.14, -3372.03,
            -2473.74, -2472.35, -2469.45,
            -1894.65, -1893.40,
            -1497.24, -1495.85, -1493.41,
            -1208.68, -1206.18, -1206.04,
            -997.92, -996.61, -996.31,
            -834.94, -834.66,
            -710.03,
            -530.16,
            -464.17
        ], dtype=float)

        self.y_data = np.array([
            0.252429,0.252141,0.251809,
            0.297989,0.296257,0.295319,
            0.339603,0.337731,0.333820,
            0.389510,0.386998,
            0.438864,0.434887,0.427893,
            0.471568,0.461699,0.461144,
            0.513532,0.506641,0.505062,
            0.535648,0.533726,
            0.568064,
            0.612886,
            0.624169
        ])

    def model(self, b, x):
        return (
            b[0]
            - b[1] * x
            - np.arctan(b[2] / (x - b[3])) / np.pi
        )

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        denom = (x - b[3])
        denom2 = denom**2

        J[:, 0] = 1.0
        J[:, 1] = -x
        J[:, 2] = -(1 / np.pi) * (1 / (1 + (b[2]/denom)**2)) * (1 / denom)
        J[:, 3] = -(1 / np.pi) * (1 / (1 + (b[2]/denom)**2)) * (b[2] / denom2)

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([0.1, -1e-5, 1000.0, -100.0]),
            "NIST Start 2": np.array([0.2, -5e-6, 1200.0, -150.0])
        }
    

class ThurberProblem(Problem):
    """
    Source: NIST
    Level of Difficulty: Higher
    Model Classification: Rational
    Number of Parameters: 7
    Number of Observations: 37
    """
    def __init__(self):
        super().__init__(name="Thurber", n=7, m=37)
        self.x_data = np.array([
            -3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699, 
            -2.633, -2.481, -2.363, -2.322, -1.501, -1.460, -1.274, -1.212, 
            -1.100, -1.046, -0.915, -0.714, -0.566, -0.545, -0.400, -0.309, 
            -0.109, -0.103, 0.010, 0.119, 0.377, 0.790, 0.963, 1.006, 
            1.115, 1.572, 1.841, 2.047, 2.200
        ])

        self.y_data = np.array([
            80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101, 
            92.405, 95.854, 100.696, 101.060, 401.672, 390.724, 567.534, 
            635.316, 733.054, 759.087, 894.206, 990.785, 1090.109, 1080.914, 
            1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543, 
            1353.863, 1414.509, 1425.208, 1421.384, 1442.962, 1464.350, 
            1468.705, 1447.894, 1457.628
        ])

    def model(self, b, x):
        """
        Rational cubic/cubic function.
        y = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3) 
        """
        numerator = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3
        denominator = 1 + b[4]*x + b[5]*x**2 + b[6]*x**3
        return numerator / denominator 

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data
    
    def J(self, b):
        x = self.x_data
        numerator = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3    
        denominator = 1 + b[4]*x + b[5]*x**2 + b[6]*x**3

        denominator_squared = denominator ** 2

        jacobian = np.zeros((self.m, self.n))

        jacobian[:, 0] = 1 / denominator
        jacobian[:, 1] = x / denominator
        jacobian[:, 2] = x**2 / denominator
        jacobian[:, 3] = x**3 / denominator

        jacobian[:, 4] = - numerator * x / denominator_squared
        jacobian[:, 5] = - numerator * x**2 / denominator_squared
        jacobian[:, 6] = - numerator * x**3 / denominator_squared

        return jacobian

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03]),
            "NIST Start 2": np.array([1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05])
        }


class Eckerle4Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Higher
    Model Classification: Exponential
    Number of Parameters: 3
    Number of Observations: 35
    """
    def __init__(self):
        super().__init__(name="Eckerle4", n=3, m=35)

        self.x_data = np.array([
            400.0,405.0,410.0,415.0,420.0,425.0,430.0,435.0,
            436.5,438.0,439.5,441.0,442.5,444.0,445.5,447.0,
            448.5,450.0,451.5,453.0,454.5,456.0,457.5,459.0,
            460.5,462.0,463.5,465.0,470.0,475.0,480.0,485.0,
            490.0,495.0,500.0
        ], dtype=float)

        self.y_data = np.array([
            0.0001575,0.0001699,0.0002350,0.0003102,0.0004917,
            0.0008710,0.0017418,0.0046400,0.0065895,0.0097302,
            0.0149002,0.0237310,0.0401683,0.0712559,0.1264458,
            0.2073413,0.2902366,0.3445623,0.3698049,0.3668534,
            0.3106727,0.2078154,0.1164354,0.0616764,0.0337200,
            0.0194023,0.0117831,0.0074357,0.0022732,0.0008800,
            0.0004579,0.0002345,0.0001586,0.0001143,0.0000710
        ])

    def model(self, b, x):
        return (b[0]/b[1]) * np.exp(-0.5 * ((x - b[2]) / b[1])**2)

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        exp_term = np.exp(-0.5 * ((x - b[2]) / b[1])**2)
        diff = (x - b[2])

        J[:, 0] = exp_term / b[1]

        J[:, 1] = -(b[0] / (b[1]**2)) * exp_term + \
                  (b[0] / b[1]) * exp_term * (diff**2 / (b[1]**3))

        J[:, 2] = (b[0] / b[1]) * exp_term * (diff / (b[1]**2))

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([1.0, 10.0, 500.0]),
            "NIST Start 2": np.array([1.5, 5.0, 450.0])
        }