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


    def _find_optimal_lambda(self, max_iter = 15, uncertainty_method = "theta_1"):
        """ Follows the Algorithm 7.3.4. """
        self.lam_L, self.lam_U, lam = self._get_initial_bounds()

        n = self.H.shape[0]
        identity = np.eye(n)
        
        # check for interior convergence ?

        for i in range(max_iter):
            # --- Step 1. Attempt to factorize H(lambda) = LL^T ---
            H_lam = self.H + lam * identity
            s, s_norm, w_norm, is_pos_def = self._perform_step(H_lam)

            in_F = is_pos_def
            in_G = False
            in_L = False
            in_N = not is_pos_def

            # set lam region L or G
            if in_F:
                if s_norm < self.delta:
                    in_G = True     # λ ∈ G
                else:
                    in_L = True     # λ ∈ L

            # --- Step 2. Update Bounds ---
            if in_G:
                self.lam_U = lam
            else:
                self.lam_L = lam

            # --- Step 3. If lambda is in F ---
            if in_F:
                # step 3a.
                lam_plus = lam + ((s_norm - self.delta)/self.delta) * (s_norm**2 / w_norm**2)
                
                # step 3b.
                if in_G:
                    # (i)
                    eigvals, eigvecs = np.linalg.eigh(H_lam)
                    u = eigvecs[:, 0]
                    h_u_val = u.T @ H_lam @ u

                    # (ii)
                    self.lam_L = max(self.lam_L, lam - h_u_val)

                    # (iii)
                    pass

                else:
                    # step 3c.
                    pass
                    # step 3d.
                    pass

            # --- Step 4. Check for termination ---
            pass

            # --- Step 5. Update lambda for the next iteration ---
            if in_L and np.linalg.norm(self.g) != 0:
                lam = lam_plus
            elif in_G:
                _, _, _, is_plus_pos_def = self._perform_step(self.H + lam_plus * identity)
                if is_plus_pos_def:
                    # step 5a
                    lam = lam_plus
                else:
                    # step 5b
                    self.lam_L = max(self.lam_L, lam_plus)
                    
                    # check lambda_L for interior concergence
                    pass
                    
                    lam = self._find_lambda_uncertainty(uncertainty_method, self.lam_L, self.lam_U)
            else:
                # lam is in N
                lam = self._find_lambda_uncertainty("theta_1", self.lam_L, self.lam_U)
        
        return lam
        
    def solve(self):
        lam_star = self._find_optimal_lambda()
        m = self.H.shape[0] 
        identity_m = np.eye(m)

        mat = lam_star * identity_m + self.H
        v = np.linalg.solve(mat, self.g)

        h_star = -(1.0 / self.M) * self.J.T @ v

        return h_star




