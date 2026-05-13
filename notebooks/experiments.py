import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from optimization_project import modified_gauss_newton


def compute_metrics(problem, x_final, rss_final):
    """
    Computes optimization quality metrics based on certified NIST reference values,
    including parameter estimation error and residual sum of squares error.
    """
    
    metrics = {}

    if problem.certified_solution is not None:

        param_error = np.linalg.norm(
            x_final - problem.certified_solution
        )

        rel_param_error = (
            param_error /
            np.linalg.norm(problem.certified_solution)
        )

        metrics["parameter_error"] = param_error
        metrics["relative_parameter_error"] = rel_param_error

    else:
        metrics["parameter_error"] = np.nan
        metrics["relative_parameter_error"] = np.nan

    if problem.certified_rss is not None:
        metrics["rss_error"] = abs(
            rss_final - problem.certified_rss
        )
        metrics["relative_rss_error"] = (
            abs(rss_final - problem.certified_rss)
            / abs(problem.certified_rss)
        )
    else:
        metrics["rss_error"] = np.nan
        metrics["relative_rss_error"] = np.nan
        
    grad_norm = np.linalg.norm(problem.J(x_final).T @ problem.F(x_final))
    metrics["gradient_norm"] = grad_norm

    return metrics


def get_problem_structure(problem):
    """
    Classifies the optimization problem structure based on the relation
    between the number of residuals and parameters.
    """
    if problem.m > problem.n:
        return "Overdetermined"

    elif problem.m < problem.n:
        return "Underdetermined"

    else:
        return "Square"
    

def run_lm(
    problem,
    x0,
    start_name,
    xtol=1e-6,
    ftol=1e-6,
    gtol=1e-6,
    max_nfev=100
):
    """
    Runs the Levenberg–Marquardt method (or TRF fallback for m < n cases)
    and returns benchmark statistics for a single starting point.
    """
    # LM requires m >= n
    method = "lm" if problem.m >= problem.n else "trf"

    result = least_squares(
        fun=problem.F,
        jac=problem.J,
        x0=x0,
        method=method,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        max_nfev=max_nfev
    )

    x_final = result.x
    rss_final = np.sum(problem.F(x_final) ** 2)

    metrics = compute_metrics(
        problem,
        x_final,
        rss_final
    )

    row = {
        "problem": problem.name,
        "difficulty": problem.difficulty,
        "classification_model": problem.classification_model,
        "structure": get_problem_structure(problem),

        "method": f"LM-{method}",
        "start_name": start_name,

        "success": result.success,

        "iterations": result.nfev,
        "final_rss": rss_final,

        "parameter_error": metrics["parameter_error"],
        "relative_parameter_error": metrics["relative_parameter_error"],
        "rss_error": metrics["rss_error"],
        "relative_rss_error": metrics["relative_rss_error"],
        "gradient_norm": metrics["gradient_norm"]
    }

    return row


def run_modified_gn(
    problem,
    x0,
    start_name,
    M0=1e-3,
    L0=1e-6,
    tol=1e-6,
    max_iter=100
):
    """
    Runs the Modified Gauss–Newton method and collects convergence
    and solution quality metrics for benchmarking purposes.
    """
    result = modified_gauss_newton(
        problem,
        x0,
        M0=M0,
        L0=L0,
        tol=tol,
        max_iter=max_iter,
    )

    x_final = result.x
    rss_final = np.sum(problem.F(x_final) ** 2)

    metrics = compute_metrics(
        problem,
        x_final,
        rss_final
    )

    row = {
        "problem": problem.name,
        "difficulty": problem.difficulty,
        "classification_model": problem.classification_model,
        "structure": get_problem_structure(problem),

        "method": "ModifiedGN",
        "start_name": start_name,

        "success": result.success,

        "iterations": result.nit,
        "final_rss": rss_final,

        "parameter_error": metrics["parameter_error"],
        "relative_parameter_error": metrics["relative_parameter_error"],
        "rss_error": metrics["rss_error"],
        "relative_rss_error": metrics["relative_rss_error"],
        "gradient_norm": metrics["gradient_norm"]
    }

    return row


def benchmark_methods(problems):
    """
    Executes benchmark experiments for all problems and starting points
    using both optimization methods and aggregates the results into a DataFrame.
    """

    rows = []

    for problem in problems:
        print(f"Running problem: {problem.name}")

        starts = problem.get_starting_points()

        for start_name, x0 in starts.items():

            row_mgn = run_modified_gn(problem, x0, start_name)
            rows.append(row_mgn)


            row_lm = run_lm(problem, x0, start_name)
            rows.append(row_lm)

    return pd.DataFrame(rows)


def run_lm_with_history(problem, x0):
    """
    Runs the Levenberg–Marquardt method while storing all evaluated parameter vectors
    to enable visualization of optimization trajectories.
    """

    x_history = []

    def wrapped_F(x):

        x_history.append(x.copy())

        return problem.F(x)

    result = least_squares(
        fun=wrapped_F,
        jac=problem.J,
        x0=x0,
        method="lm",
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=100
    )

    result.x_history = x_history

    return result


def generate_trajectory_data(
    problems,
    M0=1e-3,
    L0=1e-6,
    tol=1e-6,
    max_iter=100,
):
    """
    Generates optimization trajectory data for 2-parameter problems,
    including paths produced by LM and Modified Gauss–Newton methods.
    """

    trajectory_data = []

    for problem in problems:

        if problem.n != 2:
            continue

        starts = problem.get_starting_points()

        problem_data = {
            "problem": problem,
            "runs": []
        }

        for start_name, x0 in starts.items():

            mgn_result = modified_gauss_newton(
                problem,
                x0,
                M0=M0,
                L0=L0,
                tol=tol,
                max_iter=max_iter,
            )

            lm_result = run_lm_with_history(problem, x0)

            problem_data["runs"].append({

                "start_name": start_name,
                "x0": x0,
                "lm": lm_result,
                "modified_gn": mgn_result
            })

        trajectory_data.append(problem_data)

    return trajectory_data