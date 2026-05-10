from .problem import Problem
import numpy as np

class ThurberProblem(Problem):
    """
    Source: NIST
    Level of Difficulty: Higher
    Model Classification: Rational
    Number of Parameters: 7
    Number of Observations: 37
    """
    def __init__(self):
        super().__init__(name="Thurber", n=7, m=37,  difficulty="Higher", 
            classification_model="Rational", source="Observed",
            certified_solution=np.array([
                1.2881396800E+03,	
                1.4910792535E+03,
                5.8323836877E+02,
                7.5416644291E+01,
                9.6629502864E-01,
                3.9797285797E-01,
                4.9727297349E-02		
            ]),
            certified_rss=5.6427082397E+03
        )
        
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
        super().__init__(name="Eckerle4", n=3, m=35,  difficulty="Higher", 
            classification_model="Exponential", source="Observed",
            certified_solution=np.array([
                1.5543827178E+00,
                4.0888321754E+00,
                4.5154121844E+02
            ]),
            certified_rss=1.4635887487E-03
        )

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