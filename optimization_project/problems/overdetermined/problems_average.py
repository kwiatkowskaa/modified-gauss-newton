from ..problem import Problem
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
    

class Misra1cProblem(Problem):
    """
    Source: NIST
    Level of Difficulty: Average
    Model Classification: Miscellaneous
    Number of Parameters: 2
    Number of Observations: 14
    """

    def __init__(self):
        super().__init__(
            name="Misra1c",
            n=2,
            m=14,
            difficulty="Average",
            classification_model="Miscellaneous",
            source="NIST",
            certified_solution=np.array([
                6.3642725809E+02,
                2.0813627256E-04,
            ]),
            certified_rss=4.0966836971E-02
        )

        self.x_data = np.array([
            77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8,
            378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0
        ], dtype=float)

        self.y_data = np.array([
            10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02,
            44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78
        ])

    def model(self, b, x):
        return b[0] * (1.0 - (1.0 + 2.0 * b[1] * x) ** (-0.5))

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        denom = (1.0 + 2.0 * b[1] * x)

        J[:, 0] = 1.0 - denom ** (-0.5)
        J[:, 1] = b[0] * x * denom ** (-1.5)

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([500.0, 0.0001]),
            "NIST Start 2": np.array([600.0, 0.0002]),
        }
    

class Kirby2Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Average
    Model Classification: Rational
    Number of Parameters: 5
    Number of Observations: 151
    """

    def __init__(self):
        super().__init__(
            name="Kirby2",
            n=5,
            m=151,
            difficulty="Average",
            classification_model="Rational",
            source="NIST",
            certified_solution=np.array([
                1.6745063063E+00,
                -1.3927397867E-01,
                2.5961181191E-03,
                -1.7241811870E-03,
                2.1664802578E-05,
            ]),
            certified_rss=3.9050739624E+00
        )

        data = np.array([
            [0.0082, 9.65],
            [0.0112, 10.74],
            [0.0149, 11.81],
            [0.0198, 12.88],
            [0.0248, 14.06],
            [0.0324, 15.28],
            [0.0420, 16.63],
            [0.0549, 18.19],
            [0.0719, 19.88],
            [0.0963, 21.84],
            [0.1291, 24.00],
            [0.1710, 26.25],
            [0.2314, 28.86],
            [0.3227, 31.85],
            [0.4809, 35.79],
            [0.7084, 40.18],
            [1.0220, 44.74],
            [1.4580, 49.53],
            [1.9520, 53.94],
            [2.5410, 58.29],
            [3.2230, 62.63],
            [3.9990, 67.03],
            [4.8520, 71.25],
            [5.7320, 75.22],
            [6.7270, 79.33],
            [7.8350, 83.56],
            [9.0250, 87.75],
            [10.2670, 91.93],
            [11.5780, 96.10],
            [12.9440, 100.28],
            [14.3770, 104.46],
            [15.8560, 108.66],
            [17.3310, 112.71],
            [18.8850, 116.88],
            [20.5750, 121.33],
            [22.3200, 125.79],
            [22.3030, 125.79],
            [23.4600, 128.74],
            [24.0600, 130.27],
            [25.2720, 133.33],
            [25.8530, 134.79],
            [27.1100, 137.93],
            [27.6580, 139.33],
            [28.9240, 142.46],
            [29.5110, 143.90],
            [30.7100, 146.91],
            [31.3500, 148.51],
            [32.5200, 151.41],
            [33.2300, 153.17],
            [34.3300, 155.97],
            [35.0600, 157.76],
            [36.1700, 160.56],
            [36.8400, 162.30],
            [38.0100, 165.21],
            [38.6700, 166.90],
            [39.8700, 169.92],
            [40.0300, 170.32],
            [40.5000, 171.54],
            [41.3700, 173.79],
            [41.6700, 174.57],
            [42.3100, 176.25],
            [42.7300, 177.34],
            [43.4600, 179.19],
            [44.1400, 181.02],
            [44.5500, 182.08],
            [45.2200, 183.88],
            [45.9200, 185.75],
            [46.3000, 186.80],
            [47.0000, 188.63],
            [47.6800, 190.45],
            [48.0600, 191.48],
            [48.7400, 193.35],
            [49.4100, 195.22],
            [49.7600, 196.23],
            [50.4300, 198.05],
            [51.1100, 199.97],
            [51.5000, 201.06],
            [52.1200, 202.83],
            [52.7600, 204.69],
            [53.1800, 205.86],
            [53.7800, 207.58],
            [54.4600, 209.50],
            [54.8300, 210.65],
            [55.4000, 212.33],
            [56.4300, 215.43],
            [57.0300, 217.16],
            [58.0000, 220.21],
            [58.6100, 221.98],
            [59.5800, 225.06],
            [60.1100, 226.79],
            [61.1000, 229.92],
            [61.6500, 231.69],
            [62.5900, 234.77],
            [63.1200, 236.60],
            [64.0300, 239.63],
            [64.6200, 241.50],
            [65.4900, 244.48],
            [66.0300, 246.40],
            [66.8900, 249.35],
            [67.4200, 251.32],
            [68.2300, 254.22],
            [68.7700, 256.24],
            [69.5900, 259.11],
            [70.1100, 261.18],
            [70.8600, 264.02],
            [71.4300, 266.13],
            [72.1600, 268.94],
            [72.7000, 271.09],
            [73.4000, 273.87],
            [73.9300, 276.08],
            [74.6000, 278.83],
            [75.1600, 281.08],
            [75.8200, 283.81],
            [76.3400, 286.11],
            [76.9800, 288.81],
            [77.4800, 291.08],
            [78.0800, 293.75],
            [78.6000, 295.99],
            [79.1700, 298.64],
            [79.6200, 300.84],
            [79.8800, 302.02],
            [80.1900, 303.48],
            [80.6600, 305.65],
            [81.2200, 308.27],
            [81.6600, 310.41],
            [82.1600, 313.01],
            [82.5900, 315.12],
            [83.1400, 317.71],
            [83.5000, 319.79],
            [84.0000, 322.36],
            [84.4000, 324.42],
            [84.8900, 326.98],
            [85.2600, 329.01],
            [85.7400, 331.56],
            [86.0700, 333.56],
            [86.5400, 336.10],
            [86.8900, 338.08],
            [87.3200, 340.60],
            [87.6500, 342.57],
            [88.1000, 345.08],
            [88.4300, 347.02],
            [88.8300, 349.52],
            [89.1200, 351.44],
            [89.5400, 353.93],
            [89.8500, 355.83],
            [90.2500, 358.32],
            [90.5500, 360.20],
            [90.9300, 362.67],
            [91.2000, 364.53],
            [91.5500, 367.00],
            [92.2000, 371.30],
        ])

        self.y_data = data[:, 0]
        self.x_data = data[:, 1]

    def model(self, b, x):
        num = b[0] + b[1] * x + b[2] * x**2
        den = 1.0 + b[3] * x + b[4] * x**2
        return num / den

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        num = b[0] + b[1] * x + b[2] * x**2
        den = 1.0 + b[3] * x + b[4] * x**2

        J[:, 0] = 1.0 / den
        J[:, 1] = x / den
        J[:, 2] = x**2 / den

        J[:, 3] = -num * x / den**2
        J[:, 4] = -num * x**2 / den**2

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([2.0, -0.1, 0.003, -0.001, 0.00001]),
            "NIST Start 2": np.array([1.5, -0.15, 0.0025, -0.0015, 0.00002]),
        }