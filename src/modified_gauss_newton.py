import numpy as np
from subproblem import solve_subproblem

def modified_gauss_newton(problem, x0, M0=1.0, L0=1e-2, max_iter=100, tol=1e-6, M_search=2, 
                          return_history=False):
    
    if M_search not in [1, 2]:
        raise ValueError("M_search must be either 1 (M only grows) or 2 (M grows and shrinks).")
    
    # Initialization
    x=x0
    M=M0
    L=L0

    history = []

    for k in range(max_iter):
        F_k = problem.F(x)
        J_k = problem.J(x)

        rss = (F_k**2).sum()
        history.append(rss)

        # M search
        while True:
            h = solve_subproblem(Fx=F_k, Jx=J_k, M=M)
            candidate_point = x + h   # V_M_k <- # V_M(x_k)
            
            f_candidate_real = np.linalg.norm(problem.F(candidate_point))   # f(V_M_k(x_k))
            f_model = np.linalg.norm(F_k + J_k @ h) + 0.5 * M * np.linalg.norm(h)**2   # f_M(x_k)

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
        # TBD
        # if np.linalg.norm(problem.F(x)) < tol: <- czy to nie będzie bardziej poprawne kryterium stopu???
        # po zmianie na to kryterium w niektórych przypadkach widzę poprawę zbieżności
            print(f"Modified Gauss-Newton Converged in {k+1} iterations.")
            break  


    if return_history:
            return x, history
    return x 






