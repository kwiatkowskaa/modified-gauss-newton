
def modified_gauss_newton(problem, x0, M0=1.0, L0=1e-6, max_iter=100, tol=1e-6):
    
    # Initialization
    x=x0
    M=M0
    L=L0

    for k in range(max_iter):
        
        F_x = problem.F(x)
        JF_x = problem.JF(x)

        