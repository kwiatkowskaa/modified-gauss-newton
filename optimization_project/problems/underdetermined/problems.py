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
    

class ExtendedCraggLevyProblem(Problem):
    """
    Source: Extended Cragg and Levy problem (Luksan et al. - Problem 53)
    Level of Difficulty: N/A
    Model Classification: N/A
    Number of Parameters: n (User-defined, must be a multiple of 4)
    Number of Observations: m (User-defined, m <= n)
    """
    def __init__(self, n=200, m=5):
        if n % 4 != 0:
            raise ValueError("Number of parameters n must be a multiple of 4 due to its block structure.")
        if m > n:
            raise ValueError(f"For n={n}, the maximum number of equations m is {n}; "
                             f"otherwise, the index will go out of bounds.")
            
        super().__init__(
            name=f"ExtendedCraggLevy_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            rem = j % 4
            if rem == 0:    # mod(k,4) == 1
                F_vec[j] = (np.exp(x[j]) - x[j+1])**2
            elif rem == 1:  # mod(k,4) == 2
                F_vec[j] = 10.0 * (x[j] - x[j+1])**3
            elif rem == 2:  # mod(k,4) == 3
                F_vec[j] = np.tan(x[j] - x[j+1])**2
            elif rem == 3:  # mod(k,4) == 0
                F_vec[j] = x[j] - 1.0
                
        return F_vec

    def J(self, x):
        J_mat = np.zeros((self.m, self.n))
        
        for j in range(self.m):
            rem = j % 4
            if rem == 0:
                diff = np.exp(x[j]) - x[j+1]
                J_mat[j, j] = 2.0 * diff * np.exp(x[j])
                J_mat[j, j+1] = -2.0 * diff
            elif rem == 1:
                diff_sq = 30.0 * ((x[j] - x[j+1])**2)
                J_mat[j, j] = diff_sq
                J_mat[j, j+1] = -diff_sq
            elif rem == 2:
                tan_val = np.tan(x[j] - x[j+1])
                der = 2.0 * tan_val * (1.0 + tan_val**2)
                J_mat[j, j] = der
                J_mat[j, j+1] = -der
            elif rem == 3:
                J_mat[j, j] = 1.0
                
        return J_mat

    def get_starting_points(self):
        start1 = np.ones(self.n) * 2.0
        start1[0::4] = 1.0

        return {
            f"Luksan Standard Start (n={self.n})": start1,
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


class DiscreteBoundaryValueProblem(Problem):
    """
    Source: Discrete Boundary Value Problem (Luksan et al. - Problem 76 / Problem 213)
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
            name=f"DiscreteBVP_{n}x{m}", n=n, m=m,
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
    

class Problem46Luksan(Problem):
    """
    Source: Transcendental Block System (Luksan et al. - Problem 46)
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
            name=f"Problem46_{n}x{m}", n=n, m=m,
            difficulty="N/A",
            classification_model="N/A", 
            source="Luksan",
            certified_solution=None,
            certified_rss=None
        )

    def F(self, x):
        F_vec = np.zeros(self.m)
        
        for j in range(self.m):
            i_block = j // 5  # Integer division to find the block index
            
            # Compute the sum of cosines for the current block of 5 variables
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