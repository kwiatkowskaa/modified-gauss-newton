import numpy as np
from ..problem import Problem

class ChainedRosenbrockProblem(Problem):
    """
    Source: Chained Rosenbrock function (Luksan et al. - Problem 1)
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= 2*n - 2)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > 2 * n - 2:
            raise ValueError(f"For n={n}, the maximum number of equations m is {2*n - 2}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"ChainedRosenbrock_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            k = j // 2 
            if j % 2 == 0:  # Even equation index: 10 * (x_{i-1}^2 - x_i)
                F_vec[j] = 10.0 * (x[k]**2 - x[k+1])
            else:           # Odd equation index: x_{i-1} - 1
                F_vec[j] = x[k] - 1.0
                
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            k = j // 2 
            if j % 2 == 0:
                # Derivatives for 10 * (x_k^2 - x_{k+1})
                J_mat[j, k] = 20.0 * x[k]
                J_mat[j, k+1] = -10.0
            else:
                # Derivatives for x_k - 1
                J_mat[j, k] = 1.0
                
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n)
        start1[0::2] = -1.2
        start1[1::2] = 1.0
        
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }

class Luksan72Problem(Problem):
    """
    Source: Problem 72 (Problem 206 in [25])
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"Luksan72_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        h = 1.0 / (self.n + 1)
        h_sq = h ** 2
        
        for j in range(self.m):
            val = -2.0 * x[j] - h_sq * np.exp(x[j])
            
            if j - 1 >= 0:
                val += x[j-1]
                
            if j + 1 < self.n:
                val += x[j+1]
                
            F_vec[j] = val
            
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        h = 1.0 / (self.n + 1)
        h_sq = h ** 2
        
        for j in range(self.m):
            J_mat[j, j] = -2.0 - h_sq * np.exp(x[j])
            
            if j - 1 >= 0:
                J_mat[j, j-1] = 1.0
                
            if j + 1 < self.n:
                J_mat[j, j+1] = 1.0
                
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n)
        
        return {
            f"Standard Start (n={self.n})": start1,
        }

class TridiagonalExponentialProblem(Problem):
    """
    Source: Tridiagonal exponential problem (Luksan et al. - Problem 57)
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"TridiagonalExponential_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            k_mult = j + 1 
            
            if j == 0:
                inner_sum = x[0] + x[1]
            elif j == self.n - 1:
                inner_sum = x[j-1] + x[j]
            else:
                inner_sum = x[j-1] + x[j] + x[j+1]
                
            F_vec[j] = x[j] - np.exp(np.cos(k_mult * inner_sum))
            
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            k_mult = j + 1
            
            if j == 0:
                inner_sum = x[0] + x[1]
            elif j == self.n - 1:
                inner_sum = x[j-1] + x[j]
            else:
                inner_sum = x[j-1] + x[j] + x[j+1]
                
            # Common derivative multiplier from the chain rule
            exp_cos = np.exp(np.cos(k_mult * inner_sum))
            sin_val = np.sin(k_mult * inner_sum)
            D = k_mult * exp_cos * sin_val
            
            # Diagonal element derivative (d/dx_j)
            J_mat[j, j] = 1.0 + D
            
            # Left neighbor derivative (d/dx_{j-1})
            if j - 1 >= 0:
                J_mat[j, j-1] = D
                
            # Right neighbor derivative (d/dx_{j+1})
            if j + 1 < self.n:
                J_mat[j, j+1] = D
                
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n) * 1.5
        
        return {
            f"Luksan Standard Start (n={self.n})": start1
        }


class Luksan76Problem(Problem):
    """
    Source: Luksan et al. - Problem 76 / Problem 213
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"Luksan76_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        h = 1.0 / (self.n + 1)
        h_sq = h ** 2
        
        for j in range(self.m):
            # Base central expression
            val = 2.0 * x[j] + h_sq * (x[j] + np.sin(x[j]))
            
            # Left boundary condition: x_0 = 0
            if j - 1 >= 0:
                val -= x[j-1]
                
            # Right boundary condition: x_{n+1} = 1
            if j + 1 < self.n:
                val -= x[j+1]
            elif j + 1 == self.n:
                val -= 1.0
                
            F_vec[j] = val
            
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        h = 1.0 / (self.n + 1)
        h_sq = h ** 2
        
        for j in range(self.m):
            # Diagonal element derivative
            J_mat[j, j] = 2.0 + h_sq * (1.0 + np.cos(x[j]))
            
            # Left subdiagonal derivative (d/dx_{j-1})
            if j - 1 >= 0:
                J_mat[j, j-1] = -1.0
                
            # Right superdiagonal derivative (d/dx_{j+1})
            if j + 1 < self.n:
                J_mat[j, j+1] = -1.0
                
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n)
        
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }
    

class Luksan46Problem(Problem):
    """
    Source: Luksan et al. - Problem 46
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined, must be a multiple of 5)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n % 5 != 0:
            raise ValueError("Number of parameters n must be a multiple of 5 due to its block structure.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"Luksan46_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            i_block = j // 5 
            
            block_start = i_block * 5
            cos_sum = np.sum(np.cos(x[block_start : block_start + 5]))
            
            F_vec[j] = 5.0 - (i_block + 1.0) * (1.0 - np.cos(x[j])) - np.sin(x[j]) - cos_sum
            
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            i_block = j // 5
            block_start = i_block * 5
            
            for c in range(block_start, block_start + 5):
                if c == j:
                    J_mat[j, c] = -i_block * np.sin(x[j]) - np.cos(x[j])
                else:
                    J_mat[j, c] = np.sin(x[c])
                    
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n) * (1.0 / self.n)
        
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }


class TrigonometricExponentialProblem(Problem):
    """
    Source: Trigonometric - exponential system (Luksan et al. - Problem 47)
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"TrigExp1_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            if j == 0:
                trig_part = np.sin(x[0])**2 - np.sin(x[1])**2
                F_vec[j] = 3.0 * (x[0]**3) + 2.0 * x[1] - 5.0 + trig_part
            elif j == self.n - 1:
                F_vec[j] = 4.0 * x[j] - x[j-1] * np.exp(x[j-1] - x[j]) - 3.0
            else:
                trig_part = np.sin(x[j])**2 - np.sin(x[j+1])**2
                exp_part = x[j-1] * np.exp(x[j-1] - x[j])
                F_vec[j] = 3.0 * (x[j]**3) + 2.0 * x[j+1] - 5.0 + trig_part + 4.0 * x[j] - exp_part - 3.0
                
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            if j == 0:
                J_mat[0, 0] = 9.0 * (x[0]**2) + np.sin(2.0 * x[0])
                J_mat[0, 1] = 2.0 - np.sin(2.0 * x[1])
            elif j == self.n - 1:
                J_mat[j, j-1] = -(1.0 + x[j-1]) * np.exp(x[j-1] - x[j])
                J_mat[j, j] = 4.0 + x[j-1] * np.exp(x[j-1] - x[j])
            else:
                # Derivative w.r.t left neighbor (x_{j-1})
                J_mat[j, j-1] = -(1.0 + x[j-1]) * np.exp(x[j-1] - x[j])
                # Derivative w.r.t central node (x_j)
                J_mat[j, j] = 9.0 * (x[j]**2) + np.sin(2.0 * x[j]) + 4.0 + x[j-1] * np.exp(x[j-1] - x[j])
                # Derivative w.r.t right neighbor (x_{j+1})
                J_mat[j, j+1] = 2.0 - np.sin(2.0 * x[j+1])
                
        return J_mat

    def get_starting_points(self):
        start1 = np.zeros(self.n)

        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }
    

class ExtendedRosenbrockProblem(Problem):
    """
    Source: Extended Rosenbrock function (Luksan et al. - Problem 23)
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}.")
            
        super().__init__(
            name=f"ExtendedRosenbrock_{n}x{m}", n=n, m=m,
            difficulty="N/A", classification_model="N/A", source="Luksan",
            certified_solution=None, certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)

        for j in range(self.m):
            if j % 2 == 0:  # mod(k,2) == 1
                F_vec[j] = 10.0 * (x[j]**2 - x[j+1])
            else:           # mod(k,2) == 0
                F_vec[j] = x[j-1] - 1.0
                
        return F_vec
    
    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            if j % 2 == 0:  # mod(k,2) == 1
                J_mat[j, j] = 20.0 * x[j]
                J_mat[j, j+1] = -10.0
            else:           # mod(k,2) == 0
                J_mat[j, j-1] = 1.0
                
        return J_mat

    def get_starting_points(self):
        start1 = np.zeros(self.n)
        start1[0::2] = -1.2
        start1[1::2] = 1.0
        
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }
    

class TridiagonalSystemProblem(Problem):
    """
    Source: Tridiagonal system (Luksan et al. - Problem 40)
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}.")
            
        super().__init__(
            name=f"TridiagonalSystem_{n}x{m}", n=n, m=m,
            difficulty="N/A", classification_model="N/A", source="Luksan",
            certified_solution=None, certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            if j == 0:      # k == 1
                F_vec[j] = 4.0 * (x[0] - x[1]**2)
            elif j == self.n - 1:  # k == n
                F_vec[j] = 8.0 * x[j] * (x[j]**2 - x[j-1]) - 2.0 * (1.0 - x[j])
            else:           # 1 < k < n
                F_vec[j] = 8.0 * x[j] * (x[j]**2 - x[j-1]) - 2.0 * (1.0 - x[j]) + 4.0 * (x[j] - x[j+1]**2)
                
        return F_vec
    
    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            if j == 0:
                J_mat[0, 0] = 4.0
                J_mat[0, 1] = -8.0 * x[1]
            elif j == self.n - 1:
                J_mat[j, j-1] = -8.0 * x[j]
                J_mat[j, j] = 24.0 * (x[j]**2) - 8.0 * x[j-1] + 2.0
            else:
                J_mat[j, j-1] = -8.0 * x[j]
                J_mat[j, j] = 24.0 * (x[j]**2) - 8.0 * x[j-1] + 2.0 + 4.0
                J_mat[j, j+1] = -8.0 * x[j+1]

        return J_mat
    
    def get_starting_points(self):
        start1 = np.ones(self.n) * 12.0
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }
    

class SingularBroydenProblem(Problem):
    """
    Source: Singular Broyden problem (Luksan et al. - Problem 49)
    Number of Parameters: n (User-defined)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n < 2:
            raise ValueError("Number of parameters n must be at least 2.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}.")
            
        super().__init__(
            name=f"SingularBroyden_{n}x{m}", n=n, m=m,
            difficulty="N/A", classification_model="N/A", source="Luksan",
            certified_solution=None, certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            if j == 0:      # k == 1
                inner = (3.0 - 2.0 * x[0]) * x[0] - 2.0 * x[1] + 1.0
            elif j == self.n - 1:  # k == n
                inner = (3.0 - 2.0 * x[j]) * x[j] - x[j-1] + 1.0
            else:           # 1 < k < n
                inner = (3.0 - 2.0 * x[j]) * x[j] - x[j-1] - 2.0 * x[j+1] + 1.0
                
            F_vec[j] = inner ** 2
            
        return F_vec
    
    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            if j == 0:
                inner = (3.0 - 2.0 * x[0]) * x[0] - 2.0 * x[1] + 1.0
                # d/dx_0 [inner^2] = 2 * inner * (3 - 4*x_0)
                J_mat[0, 0] = 2.0 * inner * (3.0 - 4.0 * x[0])
                # d/dx_1 [inner^2] = 2 * inner * (-2)
                J_mat[0, 1] = -4.0 * inner
            elif j == self.n - 1:
                inner = (3.0 - 2.0 * x[j]) * x[j] - x[j-1] + 1.0
                J_mat[j, j-1] = -2.0 * inner
                J_mat[j, j] = 2.0 * inner * (3.0 - 4.0 * x[j])
            else:
                inner = (3.0 - 2.0 * x[j]) * x[j] - x[j-1] - 2.0 * x[j+1] + 1.0
                J_mat[j, j-1] = -2.0 * inner
                J_mat[j, j] = 2.0 * inner * (3.0 - 4.0 * x[j])
                J_mat[j, j+1] = -4.0 * inner
                
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n) * -1.0
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }
    

class ExtendedWoodProblem(Problem):
    """
    Source: Extended Wood problem (Luksan et al. - Problem 56)
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined, must be a multiple of 4)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n % 4 != 0:
            raise ValueError("Number of parameters n must be a multiple of 4 due to its 4-block structure.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}.")
            
        super().__init__(
            name=f"ExtendedWood_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )
    
    def F(self, x):
        F_vec = np.zeros(self.m)

        for j in range(self.m):
            if j % 4 == 0:    # mod(k,4) == 1
                F_vec[j] = -200.0 * x[j] * (x[j+1] - x[j]**2) - (1.0 - x[j])
            elif j % 4 == 1:  # mod(k,4) == 2
                F_vec[j] = 200.0 * (x[j] - x[j-1]**2) + 20.2 * (x[j] - 1.0) + 19.8 * (x[j+1] - 1.0)
            elif j % 4 == 2:  # mod(k,4) == 3
                F_vec[j] = -180.0 * x[j] * (x[j+1] - x[j]**2) - (1.0 - x[j])
            elif j % 4 == 3:  # mod(k,4) == 0
                F_vec[j] = 180.0 * (x[j] - x[j-1]**2) + 20.2 * (x[j] - 1.0) + 19.8 * (x[j-1] - 1.0)
                
        return F_vec
    
    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            if j % 4 == 0:    # mod(k,4) == 1
                J_mat[j, j] = -200.0 * x[j+1] + 600.0 * (x[j]**2) + 1.0
                J_mat[j, j+1] = -200.0 * x[j]
                
            elif j % 4 == 1:  # mod(k,4) == 2
                J_mat[j, j-1] = -400.0 * x[j-1]
                J_mat[j, j] = 200.0 + 20.2
                J_mat[j, j+1] = 19.8
                
            elif j % 4 == 2:  # mod(k,4) == 3
                J_mat[j, j] = -180.0 * x[j+1] + 540.0 * (x[j]**2) + 1.0
                J_mat[j, j+1] = -180.0 * x[j]
                
            elif j % 4 == 3:  # mod(k,4) == 0
                J_mat[j, j-1] = -360.0 * x[j-1] + 19.8
                J_mat[j, j] = 180.0 + 20.2
                
        return J_mat
    
    def get_starting_points(self):
        start1 = np.zeros(self.n)
        start1[0::2] = -3.0
        start1[1::2] = -1.0
        return {
            f"Luksan Standard Start (n={self.n})": start1,
        }