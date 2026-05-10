from .problem import Problem
import numpy as np

class MGH17Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Average
    Model Classification: Exponential
    Number of Parameters: 5
    Number of Observations: 33
    """
    def __init__(self):
        super().__init__(
            name="MGH17", n=5, m=33, difficulty="Average", 
            classification_model="Exponential", source="Generated",
            certified_solution=np.array([
                3.7541005211E-01,
                1.9358469127E+00,
                -1.4646871366E+00,
                1.2867534640E-02,
                2.2122699662E-02,
            ]),
            certified_rss=5.4648946975E-05
        )

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
        super().__init__(name="Roszman1", n=4, m=25, difficulty="Average", 
            classification_model="Miscellaneous", source="Observed",
            certified_solution=np.array([
                2.0196866396E-01,		
                -6.1953516256E-06,
                1.2044556708E+03,
                -1.8134269537E+02
            ]),
            certified_rss=4.9484847331E-04
        )

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