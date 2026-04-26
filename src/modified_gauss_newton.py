import numpy as np
from subproblem import solve_subproblem

def modified_gauss_newton(problem, x0, M0=1.0, L0=1e-6, max_iter=100, tol=1e-6, M_search=2):
    if M_search not in [1, 2]:
        raise ValueError("M_search must be either 1 (M only grows) or 2 (M grows and shrinks).")
    
    # Initialization
    x=x0
    M=M0
    L=L0

    for k in range(max_iter):
        F_k = problem.F(x)
        J_k = problem.J(x)

        # M search
        while True:
            h = solve_subproblem(Fx = F_k, Jx = J_k, M = M)
            candidate_point = x + h   # V_M_k
            
            f_candidate_real = np.linalg.norm(problem.F(candidate_point))   # f(V_M_k(x_k))
            f_model = np.linalg.norm(F_k + J_k @ h) + 0.5 * M * np.linalg.norm(h)**2   # f_M_k(x_k)

            if f_candidate_real <= f_model:
                break
            else:
                M *= 2

        x = candidate_point
        if M_search == 1:
            M = M
        elif M_search == 2:
            M = max(M * 0.5, L0)

        if np.linalg.norm(h) < tol:
            print(f"Converged in {k+1} iterations.")
            break  

    return x 






        