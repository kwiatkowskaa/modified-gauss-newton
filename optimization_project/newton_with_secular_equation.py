import numpy as np
from scipy.linalg import cholesky, solve_triangular, LinAlgError

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
        
        except LinAlgError:
            # we are in zone N of lambdas <--> cholesky functiorization is impossible
            return None, None, None, False


    def _find_optimal_lambda(self, max_iter = 15, uncertainty_method = "theta_1"):
        """ Follows the Algorithm 7.3.4. """
        self.lam_L, self.lam_U, lam = self._get_initial_bounds()

        n = self.H.shape[0]
        identity = np.eye(n)

        for i in range(max_iter):
            # --- Step 1. Attempt to factorize H(lambda) = LL^T ---
            H_lam = self.H + lam * identity
            H_lam = 0.5 * (H_lam + H_lam.T) # ensure symmetry
            
            s, s_norm, w_norm, is_pos_def = self._perform_step(H_lam)

            # check for interior convergence - 7.3.6
            if np.isclose(lam, 0.0) and is_pos_def and s_norm < self.delta:
                return 0.0

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
                    # quadratic equation
                    s_orig = s.copy()
                    sTu = np.dot(s, u)
                    square_term = np.sqrt(max(0, sTu**2 + self.delta**2 - s_norm**2))
                    
                    # two candidates for alpha
                    alpha_candidates = [-sTu + square_term, -sTu - square_term]

                    s1 = s + alpha_candidates[0] * u
                    s2 = s + alpha_candidates[1] * u
                    
                    q1 = self.g @ s1 + 0.5 * s1 @ self.H @ s1
                    q2 = self.g @ s2 + 0.5 * s2 @ self.H @ s2

                    # select alpha that minimizes model func
                    if q1 < q2:
                        alpha = alpha_candidates[0]
                        s = s1
                    else:
                        alpha = alpha_candidates[1]
                        s = s2

            # If lambda is NOT in F
            else:
                # step 3c.
                # Cholesky factorization will encounter a nonpositive pivot at the kth stage of the decomposition
                L, k, d_kk = self._partial_cholesky(H_lam)
                    
                delta = -d_kk

                # find vector v such that (H + delta*e_k e_k^T) v = 0
                n = L.shape[0]
                v = np.zeros(n)

                v[k] = 1.0

                for j in range(k - 1, -1, -1):
                    tmp = np.dot(L[j + 1:k + 1, j], v[j + 1:k + 1])
                    v[j] = -tmp / L[j, j]    

                # step 3d.
                lambda_B = lam + delta / (np.linalg.norm(v) ** 2)
                self.lam_L = max(self.lam_L, lambda_B)


            # --- Step 4. Check for termination ---
            """ Algorithm 7.3.5 """
            k_easy = 0.1
            k_hard = 0.2

            # EASY CASE
            if in_F and abs(s_norm - self.delta) <= k_easy * self.delta:
                break

            if in_G and np.isclose(lam, 0.0):
                break
                
            # HARD CASE
            if in_G and alpha**2 * (u.T @ H_lam @ u) <= k_hard * (s_orig.T @ H_lam @ s_orig + lam * self.delta**2):
                break


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

        try:
            v = np.linalg.solve(mat, self.g)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(mat, self.g, rcond=1e-16)[0]

        h_star = -(1.0 / self.M) * self.J.T @ v

        return h_star



    def _partial_cholesky(self, A):
        """
        Partial Cholesky factorization.
        """
        n = A.shape[0]
        L = np.zeros_like(A)

        min_d_kk = float('inf')
        min_k = 0

        for k in range(n):
            s = np.sum(L[k, :k] ** 2)
            d_kk = A[k, k] - s

            if d_kk < min_d_kk:
                min_d_kk = d_kk
                min_k = k

            if d_kk <= 1e-16:
                return L, k, d_kk
            
            L[k, k] = np.sqrt(d_kk)

            for i in range(k + 1, n):
                s2 = np.sum(L[i, :k] * L[k, :k])
                L[i, k] = (A[i, k] - s2) / L[k, k]

        return L, min_k, min_d_kk




