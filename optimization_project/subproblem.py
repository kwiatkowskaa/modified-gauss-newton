import numpy as np

def solve_subproblem(Fx, Jx, M, tol=1e-8):
    
    m = Fx.shape[0]
    n = Jx.shape[1]

    Q1 = np.eye(n)
    Q2 = np.eye(m)

    Q1_inv = np.linalg.inv(Q1)
    A = (1.0 / M) * (Jx @ Q1_inv @ Jx.T)

    
    # Find optimal lambda using bisection method
    def phi_prime(lam):
        mat = lam * Q2 + A
        v = np.linalg.solve(mat, Fx)
        return 0.5 - 0.5 * (v @ (Q2 @ v))

    lam_low = 0.0
    lam_high = 1.0

    while phi_prime(lam_high) < 0:
        lam_high *= 2

    for _ in range(50):
        lam_mid = 0.5 * (lam_low + lam_high)
        val = phi_prime(lam_mid)

        if abs(val) < tol:
            lam = lam_mid
            break

        if val > 0:
            lam_high = lam_mid
        else:
            lam_low = lam_mid
    else:
        lam = lam_mid


    # Step (39) from article
    mat = lam * Q2 + A
    v = np.linalg.solve(mat, Fx)

    h = -(1.0 / M) * Q1_inv @ Jx.T @ v

    return h