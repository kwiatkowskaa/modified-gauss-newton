import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

difficulty_order = ["Lower", "Average", "Higher"]

method_palette = {
    "ModifiedGN": "#d81159",
    "LM-lm": "#1b9aaa"
}


def plot_boxplot(df, x, y, xlabel, ylabel, order=None, palette=None):
    """
    Creates a boxplot comparing optimization metrics across problem groups and methods.
    Uses logarithmic scaling on the y-axis to improve visibility of metric distributions.
    """

    plt.figure(figsize=(10,4))

    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue="method",
        order=order,
        palette=palette,
        gap=0.1
    )

    plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (log scale)")
    plt.title(f"{ylabel} vs {xlabel} and optimization method")

    plt.legend(title="Method")

    plt.grid(alpha=0.3, axis="y")

    plt.show()



def plot_optimizer_paths(
    problem_data,
    grid_size=200,
    padding=0.2,
):
    """
    Visualizes optimization trajectories of LM and Modified Gauss–Newton methods
    on a contour map of the residual sum of squares landscape for 2-parameter problems.
    """

    problem = problem_data["problem"]
    runs = problem_data["runs"]
    n_plots = len(runs)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), constrained_layout=True)

    if n_plots == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):

        start_name = run["start_name"]
        x0 = run["x0"]
        lm_res = run["lm"]
        mgn_res = run["modified_gn"]

        # ----------------------------------------
        # collect all points
        # ----------------------------------------

        all_points = np.vstack([

            np.array(lm_res.x_history),
            np.array(mgn_res.x_history),
            problem.certified_solution.reshape(1, -1)
        ])

        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)

        dx = x_max - x_min
        dy = y_max - y_min

        x_min -= padding * dx
        x_max += padding * dx

        y_min -= padding * dy
        y_max += padding * dy

        # ----------------------------------------
        # contour background
        # ----------------------------------------

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size)
        )

        Z = np.zeros_like(xx)

        for i in range(grid_size):
            for j in range(grid_size):

                p = np.array([
                    xx[i, j],
                    yy[i, j]
                ])

                try:
                    Z[i, j] = np.sum(problem.F(p)**2)

                except:
                    Z[i, j] = np.nan

        contour = ax.contourf(
            xx,
            yy,
            np.log10(Z + 1e-16),
            levels=50,
            cmap="Greys_r"
        )

        # ----------------------------------------
        # LM path
        # ----------------------------------------

        lm_path = np.array(lm_res.x_history)

        ax.plot(
            lm_path[:, 0],
            lm_path[:, 1],
            '-o',
            markersize=6,
            linewidth=2,
            label='LM',
            color=method_palette["LM-lm"],
        )

        # ----------------------------------------
        # Modified GN path
        # ----------------------------------------

        mgn_path = np.array(mgn_res.x_history)

        ax.plot(
            mgn_path[:, 0],
            mgn_path[:, 1],
            '-s',
            markersize=6,
            linewidth=2,
            label='Modified GN',
            color=method_palette["ModifiedGN"],
        )

        # ----------------------------------------
        # starting point
        # ----------------------------------------

        ax.scatter(
            x0[0],
            x0[1],
            marker='x',
            s=100,
            linewidths=3,
            label='Start',
            color="black"
        )

        if problem.certified_solution is not None:

            x_star = problem.certified_solution

            ax.scatter(
                x_star[0],
                x_star[1],
                marker='*',
                s=260,
                label='Certified',
                color='#d81159'
            )

        ax.set_title(start_name)
        ax.set_xlabel(r'$\beta_1$')
        ax.set_ylabel(r'$\beta_2$')
        ax.legend()

    
    fig.suptitle(f"{problem.name} optimization trajectories")

    plt.colorbar(
        contour,
        ax=axes,
        label=r'$\log_{10}(RSS)$'
    )

    plt.show()