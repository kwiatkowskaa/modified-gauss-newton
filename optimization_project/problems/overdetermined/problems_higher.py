from ..problem import Problem
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
    

class BoxBODProblem(Problem):
    """
    Source: NIST/ITL StRD
    Level of Difficulty: Higher
    Model Classification: Exponential
    Number of Parameters: 2
    Number of Observations: 6
    """
    def __init__(self):
        super().__init__(
            name="BoxBOD", n=2, m=6, difficulty="Higher",
            classification_model="Exponential", source="NIST/ITL StRD",
            certified_solution=np.array([
                2.1380940889E+02,
                5.4723748542E-01
            ]),
            certified_rss=1.1680088766E+03
        )

        self.x_data = np.array([
            1,
            2,
            3,
            5,
            7,
            10
        ], dtype=float)

        self.y_data = np.array([
            109,
            149,
            149,
            191,
            213,
            224
        ], dtype=float)

    def model(self, b, x):
        return b[0] * (1 - np.exp(-b[1] * x))

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        exp_term = np.exp(-b[1] * x)

        J[:, 0] = (1 - exp_term)
        J[:, 1] = b[0] * (x * exp_term)

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([1.0, 1.0]),
            "NIST Start 2": np.array([100.0, 0.75])
        }
    

class Rat42Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Higher
    Model Classification: Exponential
    Number of Parameters: 3
    Number of Observations: 9
    """

    def __init__(self):
        super().__init__(
            name="Rat42",
            n=3,
            m=9,
            difficulty="Higher",
            classification_model="Exponential",
            source="NIST",
            certified_solution=np.array([
                7.2462237576E+01,
                2.6180768402E+00,
                6.7359200066E-02,
            ]),
            certified_rss=8.0565229338E+00
        )

        self.x_data = np.array([
            9.0, 14.0, 21.0, 28.0, 42.0,
            57.0, 63.0, 70.0, 79.0
        ], dtype=float)

        self.y_data = np.array([
            8.93, 10.80, 18.59, 22.33, 39.35,
            56.11, 61.73, 64.62, 67.08
        ])

    def model(self, b, x):
        return b[0] / (1.0 + np.exp(b[1] - b[2] * x))

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        exp_term = np.exp(b[1] - b[2] * x)
        denom = (1.0 + exp_term)

        J[:, 0] = 1.0 / denom
        J[:, 1] = -b[0] * exp_term / denom**2
        J[:, 2] = b[0] * x * exp_term / denom**2

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([100.0, 1.0, 0.1]),
            "NIST Start 2": np.array([75.0, 2.5, 0.07]),
        }
    

class Bennett5Problem(Problem):
    """
    Source: NIST
    Level of Difficulty: Higher
    Model Classification: Miscellaneous
    Number of Parameters: 3
    Number of Observations: 154
    """

    def __init__(self):
        super().__init__(
            name="Bennett5",
            n=3,
            m=154,
            difficulty="Higher",
            classification_model="Miscellaneous",
            source="NIST",
            certified_solution=np.array([
                -2.5235058043E+03,
                4.6736564644E+01,
                9.3218483193E-01,
            ]),
            certified_rss=5.2404744073E-04
        )

        data = np.array([
            [-34.834702,  7.447168],
            [-34.393200,  8.102586],
            [-34.152901,  8.452547],
            [-33.979099,  8.711278],
            [-33.845901,  8.916774],
            [-33.732899,  9.087155],
            [-33.640301,  9.232590],
            [-33.559200,  9.359535],
            [-33.486801,  9.472166],
            [-33.423100,  9.573384],
            [-33.365101,  9.665293],
            [-33.313000,  9.749461],
            [-33.260899,  9.827092],
            [-33.217400,  9.899128],
            [-33.176899,  9.966321],
            [-33.139198, 10.029280],
            [-33.101601, 10.088510],
            [-33.066799, 10.144430],
            [-33.035000, 10.197380],
            [-33.003101, 10.247670],
            [-32.971298, 10.295560],
            [-32.942299, 10.341250],
            [-32.916302, 10.384950],
            [-32.890202, 10.426820],
            [-32.864101, 10.467000],
            [-32.841000, 10.505640],
            [-32.817799, 10.542830],
            [-32.797501, 10.578690],
            [-32.774300, 10.613310],
            [-32.757000, 10.646780],
            [-32.733799, 10.679150],
            [-32.716400, 10.710520],
            [-32.699100, 10.740920],
            [-32.678799, 10.770440],
            [-32.661400, 10.799100],
            [-32.644001, 10.826970],
            [-32.626701, 10.854080],
            [-32.612202, 10.880470],
            [-32.597698, 10.906190],
            [-32.583199, 10.931260],
            [-32.568699, 10.955720],
            [-32.554298, 10.979590],
            [-32.539799, 11.002910],
            [-32.525299, 11.025700],
            [-32.510799, 11.047980],
            [-32.499199, 11.069770],
            [-32.487598, 11.091100],
            [-32.473202, 11.111980],
            [-32.461601, 11.132440],
            [-32.435501, 11.152480],
            [-32.435501, 11.172130],
            [-32.426800, 11.191410],
            [-32.412300, 11.210310],
            [-32.400799, 11.228870],
            [-32.392101, 11.247090],
            [-32.380501, 11.264980],
            [-32.366001, 11.282560],
            [-32.357300, 11.299840],
            [-32.348598, 11.316820],
            [-32.339901, 11.333520],
            [-32.328400, 11.349940],
            [-32.319698, 11.366100],
            [-32.311001, 11.382000],
            [-32.299400, 11.397660],
            [-32.290699, 11.413070],
            [-32.282001, 11.428240],
            [-32.273300, 11.443200],
            [-32.264599, 11.457930],
            [-32.256001, 11.472440],
            [-32.247299, 11.486750],
            [-32.238602, 11.500860],
            [-32.229900, 11.514770],
            [-32.224098, 11.528490],
            [-32.215401, 11.542020],
            [-32.203800, 11.555380],
            [-32.198002, 11.568550],
            [-32.189400, 11.581560],
            [-32.183601, 11.594420],
            [-32.174900, 11.607121],
            [-32.169102, 11.619640],
            [-32.163300, 11.632000],
            [-32.154598, 11.644210],
            [-32.145901, 11.656280],
            [-32.140099, 11.668200],
            [-32.131401, 11.679980],
            [-32.125599, 11.691620],
            [-32.119801, 11.703130],
            [-32.111198, 11.714510],
            [-32.105400, 11.725760],
            [-32.096699, 11.736880],
            [-32.090900, 11.747890],
            [-32.088001, 11.758780],
            [-32.079300, 11.769550],
            [-32.073502, 11.780200],
            [-32.067699, 11.790730],
            [-32.061901, 11.801160],
            [-32.056099, 11.811480],
            [-32.050301, 11.821700],
            [-32.044498, 11.831810],
            [-32.038799, 11.841820],
            [-32.033001, 11.851730],
            [-32.027199, 11.861550],
            [-32.024300, 11.871270],
            [-32.018501, 11.880890],
            [-32.012699, 11.890420],
            [-32.004002, 11.899870],
            [-32.001099, 11.909220],
            [-31.995300, 11.918490],
            [-31.989500, 11.927680],
            [-31.983700, 11.936780],
            [-31.977900, 11.945790],
            [-31.972099, 11.954730],
            [-31.969299, 11.963590],
            [-31.963501, 11.972370],
            [-31.957701, 11.981070],
            [-31.951900, 11.989700],
            [-31.946100, 11.998260],
            [-31.940300, 12.006740],
            [-31.937401, 12.015150],
            [-31.931601, 12.023490],
            [-31.925800, 12.031760],
            [-31.922899, 12.039970],
            [-31.917101, 12.048100],
            [-31.911301, 12.056170],
            [-31.908400, 12.064180],
            [-31.902599, 12.072120],
            [-31.896900, 12.080010],
            [-31.893999, 12.087820],
            [-31.888201, 12.095580],
            [-31.885300, 12.103280],
            [-31.882401, 12.110920],
            [-31.876600, 12.118500],
            [-31.873699, 12.126030],
            [-31.867901, 12.133500],
            [-31.862101, 12.140910],
            [-31.859200, 12.148270],
            [-31.856300, 12.155570],
            [-31.850500, 12.162830],
            [-31.844700, 12.170030],
            [-31.841801, 12.177170],
            [-31.838900, 12.184270],
            [-31.833099, 12.191320],
            [-31.830200, 12.198320],
            [-31.827299, 12.205270],
            [-31.821600, 12.212170],
            [-31.818701, 12.219030],
            [-31.812901, 12.225840],
            [-31.809999, 12.232600],
            [-31.807100, 12.239320],
            [-31.801300, 12.245990],
            [-31.798401, 12.252620],
            [-31.795500, 12.259200],
            [-31.789700, 12.265750],
            [-31.786800, 12.272240],
        ])

        self.y_data = data[:, 0]
        self.x_data = data[:, 1]

    def model(self, b, x):
        return b[0] * (b[1] + x) ** (-1.0 / b[2])

    def F(self, b):
        return self.model(b, self.x_data) - self.y_data

    def J(self, b):
        x = self.x_data
        J = np.zeros((self.m, self.n))

        base = b[1] + x
        power = -1.0 / b[2]

        J[:, 0] = base ** power

        J[:, 1] = (
            b[0]
            * power
            * base ** (power - 1.0)
        )

        J[:, 2] = (
            b[0]
            * base ** power
            * np.log(base)
            / (b[2] ** 2)
        )

        return J

    def get_starting_points(self):
        return {
            "NIST Start 1": np.array([-2000.0, 50.0, 0.8]),
            "NIST Start 2": np.array([-1500.0, 45.0, 0.85]),
        }