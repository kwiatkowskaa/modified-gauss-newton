import numpy as np
from scipy.optimize import OptimizeResult

from .newton_with_secular_equation import NewtonSecularEquationSolver

def modified_gauss_newton(problem, x0, M0=1e-3, L0=1e-6, max_iter=100, tol=1e-6):
    
    # Initialization
    x=x0
    M=M0
    L=L0

    rss_history = []
    x_history = []

    success = False

    for k in range(max_iter):
        F_k = problem.F(x)
        J_k = problem.J(x)

        rss = (F_k**2).sum()
        rss_history.append(rss)
        x_history.append(x.copy())

        # M search
        while True:

            solver = NewtonSecularEquationSolver(g=F_k, J=J_k, M=M, delta=1)
            h = solver.solve()

            candidate_point = x + h   # V_M_k <- # V_M(x_k)
            
            f_candidate_real = np.linalg.norm(problem.F(candidate_point))   # f(V_M_k(x_k))
            f_model = np.linalg.norm(F_k + J_k @ h) + 0.5 * M * np.linalg.norm(h)**2   # f_M(x_k)

            if f_candidate_real <= f_model:
                break
            else:
                M *= 2

        x = candidate_point

        M = max(M * 0.5, L0)

        if np.linalg.norm(h) < tol: # step convergence
            success = True
            break

        if np.linalg.norm(F_k) < tol: # function convergence
            success = True
            break
    
    F_final = problem.F(x)
    final_rss = np.sum(F_final**2)

    return OptimizeResult(
        x=x,
        x_history=x_history,
        fun=F_final,

        success=success,

        nit=k+1,

        rss=final_rss,
        rss_history=rss_history
    )






