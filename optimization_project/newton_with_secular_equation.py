import numpy as np
from scipy.linalg import cholesky, solve_triangular

class NewtonSecularEquationSolver:
    """
    Assume Q1 = Q2 = I
    """
    def __init__(self, g, J, M, delta):
        self.g = g
        self.J = J
        self.M = M
        self.delta = delta

        self.H = (1.0 / self.M) * (self.J @ self.J.T)

        self.lam_L = None
        self.lam_U = None
        self.lam_start = None

    def _get_initial_bounds(self):
        """ (7.3.8 section) - page 192 """
        g_norm = np.linalg.norm(self.g)
        diag_H = np.diag(self.H)
        off_diag_abs_sums_H = np.sum(np.abs(self.H), axis=1) - np.abs(diag_H)

        H_frob_norm = np.linalg.norm(self.H, ord='fro')
        H_inf_norm = np.linalg.norm(self.H, ord=np.inf)

        # 1. Calculate lambda_L (Lower Bound)
        h_max_gersh = np.max(diag_H + off_diag_abs_sums_H)
        m_max = min(h_max_gersh, H_frob_norm, H_inf_norm)
        self.lam_L = max(0, -np.min(diag_H), g_norm / self.delta - m_max)

        # 2. Calculate lambda_U (Upper Bound)
        h_max_gersh = np.max(-diag_H + off_diag_abs_sums_H)
        m_max = min(h_max_gersh, H_frob_norm, H_inf_norm)
        self.lam_U = max(0, g_norm / self.delta + m_max)

        # 3. Suggested starting value for lambda
        if self.lam_L == 0:
            self.lam_start = 0.0
        else:
            self.lam_start = max(self.lam_L, g_norm / self.delta)
            
        return  self.lam_L, self.lam_U, self.lam_start
    
    def _find_lambda_uncertainty(self, method, lam_L, lam_U):
        """ (7.3.6 section) - page 189 """
        geo_mean = np.sqrt(lam_L * lam_U) 
        theta = 0.01

        if method == "arithmetic_mean":
            return 0.5 * (lam_L + lam_U)
        
        elif method == "geometric_mean":
            return geo_mean
        
        elif method == "theta_1":
            return max(geo_mean, lam_L+theta*(lam_U-lam_L))
        
        elif method == "theta_2":
            theta = 0.01
            return max(geo_mean, theta*lam_U)
        
        raise ValueError(f"Unknown method: {method}")
    
    def _perform_step(self, H_lam):
        """ Algorithm 7.3.1 """
        # case: lam > - lam_1
        try:
            # we are in zone F of lambdas
            L = cholesky(H_lam, lower=True)

            # solve LL^T * s = -g
            y = solve_triangular(L, -self.g, lower=True)
            s = solve_triangular(L.T, y, lower=False)
            s_norm = np.linalg.norm(s)

            # solve (L*w = s)
            w = solve_triangular(L, s, lower=True)
            w_norm = np.linalg.norm(w)

            return s, s_norm, w_norm, True
        
        except np.linalg.LinAlgError:
            # we are in zone N of lambdas <--> cholesky functiorization is impossible
            return None, None, None, False


    # def solve(self):
    #     lam_L, lam_U, lam = self._get_initial_bounds()

    #     n = self.H.shape[0]
    #     identity = np.eye(n)

    #     # STEP 1: check for INTERIOR SOLUTION - page 174
    #     s, s_norm, w_norm, is_pos_def = self._perform_step(self.H)

    #     if is_pos_def and s_norm < self.delta:
    #         # case lam is in G region
    #         return 0.0, s







