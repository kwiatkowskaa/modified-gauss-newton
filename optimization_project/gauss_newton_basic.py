import numpy as np

def gauss_newton_basic(problem, x0, tol=1e-8, max_iter=100, return_history=False):
    x = x0.copy()
    history = []

    for i in range(max_iter):
        F = problem.F(x)
        J = problem.J(x)

        history.append(float(np.sum(F ** 2)))

        try:
            delta_x, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(delta_x) < tol:
            print(f"Basic Gauss-Newton converged in {i} iterations.")
            break

        x = x + delta_x

    history.append(float(np.sum(problem.F(x) ** 2)))

    if return_history:
        return x, history
    return x