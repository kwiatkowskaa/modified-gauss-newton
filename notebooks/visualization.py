import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from experiments import run_lm_with_history
from optimization_project.modified_gauss_newton import modified_gauss_newton

difficulty_order = ["Lower", "Average", "Higher"]

method_palette = {
    "ModifiedGN": "#cf0c43",
    "LM-lm": "#2F52E0", 
    "LM-trf": "#FFC857"
}


def plot_boxplot(df, x, y, xlabel, ylabel, order=None, palette=None):
    """
    Creates a boxplot comparing optimization metrics across problem groups and methods.
    Uses logarithmic scaling on the y-axis to improve visibility of metric distributions.
    """

    plt.figure(figsize=(9, 4))

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
    plt.title(f"{ylabel} vs {xlabel} and Optimization Method")

    plt.legend(title="Method")
    plt.grid(alpha=0.3, axis="y")
    plt.show()


def plot_boxplot_with_connections(df, x, y, xlabel, ylabel, order=None, palette=None):
    """
    Creates an enhanced boxplot comparing optimization metrics across groups and methods.
    Overlays individual dataset points and connects paired results (same dataset run, 
    different methods) with lines to clearly visualize the performance shift.
    """
    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    categories = order if order is not None else sorted(df[x].unique())
    methods = sorted(df['method'].unique())
    
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue="method",
        order=categories,
        hue_order=methods,
        palette=palette,
        gap=0.1,
        boxprops=dict(alpha=0.6),
        showfliers=False,
        ax=ax
    )

    all_x_coords = [np.mean(line.get_xdata()) for line in ax.lines]
    unique_x_coords = sorted(list(set(np.round(all_x_coords, 4))))
    
    x_pos_map = {}
    idx = 0
    for cat in categories:
        x_pos_map[cat] = {}
        for mth in methods:
            if idx < len(unique_x_coords):
                x_pos_map[cat][mth] = unique_x_coords[idx]
                idx += 1

    df_paired = df.copy()
    df_paired['run_id'] = df_paired['problem'] + " | " + df_paired['start_name']
    
    for cat in categories:
        df_cat = df_paired[df_paired[x] == cat]
        unique_runs = df_cat['run_id'].unique()
        
        for run in unique_runs:
            df_run = df_cat[df_cat['run_id'] == run]
            
            if len(df_run) == 2:
                row1 = df_run.iloc[0]
                row2 = df_run.iloc[1]
                
                m1, m2 = row1['method'], row2['method']
                y1, y2 = row1[y], row2[y]
                
                x1 = x_pos_map[cat].get(m1)
                x2 = x_pos_map[cat].get(m2)
                
                if x1 is not None and x2 is not None:
                    ax.plot([x1, x2], [y1, y2], color='black', alpha=0.5, linestyle='-', linewidth=0.8, zorder=2)
                
                    c1 = palette[m1]
                    c2 = palette[m2]
                    
                    ax.scatter(x1, y1, color=c1, s=35, edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)
                    ax.scatter(x2, y2, color=c2, s=35, edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)
            
            elif len(df_run) == 1:
                row = df_run.iloc[0]
                mth = row['method']
                y_val = row[y]
                x_val = x_pos_map[cat].get(mth)
                if x_val is not None:
                    c = palette[mth]
                    ax.scatter(x_val, y_val, color=c, s=35, edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)

    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (log scale)")
    plt.title(f"{ylabel} vs {xlabel} with Paired Method Connections")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
            
    plt.legend(unique_handles, unique_labels, title="Method")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()


    """
    Generates a 5x5 matrix of convergence subplots (RSS vs Iterations) 
    for a given problem class.
    Compares the Modified Gauss-Newton (Secular) approach against SciPy Least Squares.
    Prints mean and standard deviation for final RSS and iteration counts.
    """
    if len(n_values) != 5 or len(m_values) != 5:
        raise ValueError("Both n_values and m_values must contain exactly 5 elements.")
        
    if method_palette is None:
        raise ValueError("Provide palette of colours.")

    fig, axes = plt.subplots(5, 5, figsize=(12, 9), sharex=True, sharey=True, dpi=300)
    
    print(f"Initializing enhanced 5x5 convergence matrix for {problem_cls.__name__}...")
    
    legend_handles = None
    
    mgn_final_rss = []
    mgn_iterations = []
    
    scipy_final_rss = []
    scipy_iterations = []
    
    for i, n in enumerate(n_values):
        print(f"Processing matrix row {i+1}/5 (n = {n})...")
        for j, m in enumerate(m_values):
            ax = axes[i, j]
            
            try:
                problem = problem_cls(n=n, m=m)
                starts = problem.get_starting_points()
                start_name = list(starts.keys())[0]
                x0 = starts[start_name]
                valid_geometry = True
            except Exception:
                # Gray out cells that violate structural problem limits
                ax.set_facecolor('#d9d9d9') 
                ax.text(0.5, 0.5, f"Invalid\nGeometry\nn={n}\nm={m}", 
                        color='#555555',
                        ha='center', va='center', transform=ax.transAxes)
                valid_geometry = False
                
            if valid_geometry:
                if m < n:
                    ax.set_facecolor('#e6f2ff')
                else:
                    ax.set_facecolor('#fff9f0')

                # ---------------------------------------------------
                # 1. Modified Gauss-Newton (Secular)
                # ---------------------------------------------------
                res_sec = modified_gauss_newton(
                    problem, x0, M0=1e-3, L0=1e-6, tol=1e-6, max_iter=100
                )
                
                final_rss_mgn = res_sec.rss_history[-1]
                iters_mgn = len(res_sec.rss_history) - 1
                
                mgn_final_rss.append(final_rss_mgn)
                mgn_iterations.append(iters_mgn)
                
                # ---------------------------------------------------
                # 2. SciPy Least Squares 
                # ---------------------------------------------------
                x_history = []
                def wrapped_F(x):
                    x_history.append(x.copy())
                    return problem.F(x)
                
                scipy_method = "lm" if problem.m >= problem.n else "trf"
                try:
                    res_sp = least_squares(
                        fun=wrapped_F, jac=problem.J, x0=x0, method=scipy_method,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=100
                    )
                    hist_sp = [np.sum(problem.F(x) ** 2) for x in x_history]
                    
                    iters_sp = res_sp.niter if hasattr(res_sp, 'niter') else len(hist_sp) - 1
                    final_rss_sp = hist_sp[-1]
                    
                    scipy_final_rss.append(final_rss_sp)
                    scipy_iterations.append(iters_sp)
                except Exception:
                    hist_sp = [] 
                
                rss_sec = np.clip(res_sec.rss_history, 1e-32, None)
                rss_sp = np.clip(hist_sp, 1e-32, None)
                
                line_sec, = ax.semilogy(rss_sec, color=method_palette.get("ModifiedGN"), linewidth=1.5)
                
                if len(hist_sp) > 0:
                    line_sp, = ax.semilogy(rss_sp, color=method_palette.get("LM-lm"), linewidth=1.5)
                    if legend_handles is None:
                        legend_handles = [line_sec, line_sp]
                
                ax.text(0.95, 0.95, f"n: {n}\nm: {m}", color='black',
                        ha='right', va='top', transform=ax.transAxes, 
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))
                
                ax.grid(True, which="both", ls="-", alpha=0.2)
            
            ax.tick_params(axis='both', which='major')
            
            if i == 4:
                ax.set_xlabel("Iteration", fontsize=12)
            if j == 0:
                ax.set_ylabel("$\|F(x)\|^2$", fontsize=12)

    if legend_handles is not None:
        fig.legend(
            legend_handles, 
            ["Modified GN", "SciPy"],
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.95),
            ncol=2, 
            fontsize=12,
            frameon=True,
        )
    
    fig.suptitle(f"Convergence Matrix (5x5) for {problem_cls.__name__}", y=0.98, fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    
    print("\n" + "="*60)
    print(f" STATISTICAL SUMMARY FOR {problem_cls.__name__.upper()} ")
    print("="*60)
    
    if len(mgn_final_rss) > 0:
        print(f"{'Metric':<20} | {'Modified GN':<18} | {'SciPy':<18}")
        print("-"*60)
        
        mean_rss_mgn, std_rss_mgn = np.mean(mgn_final_rss), np.std(mgn_final_rss)
        mean_rss_sp, std_rss_sp = np.mean(scipy_final_rss), np.std(scipy_final_rss) if scipy_final_rss else (0.0, 0.0)
        print(f"{'Mean Final $\|F(x)\|^2$':<20} | {mean_rss_mgn:<18.4e} | {mean_rss_sp:<18.4e}")
        print(f"{'Std Final $\|F(x)\|^2$':<20} | {std_rss_mgn:<18.4e} | {std_rss_sp:<18.4e}")
        print("-"*60)
        
        mean_iter_mgn, std_iter_mgn = np.mean(mgn_iterations), np.std(mgn_iterations)
        mean_iter_sp, std_iter_sp = np.mean(scipy_iterations), np.std(scipy_iterations) if scipy_iterations else (0.0, 0.0)
        print(f"{'Mean Iterations':<20} | {mean_iter_mgn:<18.2f} | {mean_iter_sp:<18.2f}")
        print(f"{'Std Iterations':<20} | {std_iter_mgn:<18.2f} | {std_iter_sp:<18.2f}")
    else:
        print("No valid configurations were evaluated to generate statistics.")
    print("="*60 + "\n")

def plot_convergence_grid_5x5(problem_cls, n_values, m_values, method_palette=None):
    """
    Generates a 5x5 matrix of convergence subplots (RSS vs Iterations) 
    for a given problem class.
    Compares the Modified Gauss-Newton (Secular) approach against SciPy Least Squares.
    Prints robust mean and standard deviation for final RSS and iteration counts.
    """
    if len(n_values) != 5 or len(m_values) != 5:
        raise ValueError("Both n_values and m_values must contain exactly 5 elements.")
        
    if method_palette is None:
        raise ValueError("Provide palette of colours.")

    fig, axes = plt.subplots(5, 5, figsize=(12, 9), sharex=True, sharey=True, dpi=300)
    
    print(f"Initializing enhanced 5x5 convergence matrix for {problem_cls.__name__}...")
    
    legend_handles = None
    
    mgn_final_rss = []
    mgn_iterations = []
    
    scipy_final_rss = []
    scipy_iterations = []
    
    for i, n in enumerate(n_values):
        print(f"Processing matrix row {i+1}/5 (n = {n})...")
        for j, m in enumerate(m_values):
            ax = axes[i, j]
            
            try:
                problem = problem_cls(n=n, m=m)
                starts = problem.get_starting_points()
                start_name = list(starts.keys())[0]
                x0 = starts[start_name]
                valid_geometry = True
            except Exception:
                # Gray out cells that violate structural problem limits
                ax.set_facecolor('#d9d9d9') 
                ax.text(0.5, 0.5, f"Invalid\nGeometry\nn={n}\nm={m}", 
                        color='#555555',
                        ha='center', va='center', transform=ax.transAxes)
                valid_geometry = False
                
            if valid_geometry:
                if m < n:
                    ax.set_facecolor('#e6f2ff')
                else:
                    ax.set_facecolor('#fff9f0')

                # ---------------------------------------------------
                # 1. Modified Gauss-Newton (Secular)
                # ---------------------------------------------------
                res_sec = modified_gauss_newton(
                    problem, x0, M0=1e-3, L0=1e-6, tol=1e-6, max_iter=100
                )
                
                final_rss_mgn = res_sec.rss_history[-1]
                iters_mgn = len(res_sec.rss_history) - 1
                
                mgn_final_rss.append(final_rss_mgn)
                mgn_iterations.append(iters_mgn)
                
                # ---------------------------------------------------
                # 2. SciPy Least Squares 
                # ---------------------------------------------------
                x_history = []
                def wrapped_F(x):
                    x_history.append(x.copy())
                    return problem.F(x)
                
                scipy_method = "lm" if problem.m >= problem.n else "trf"
                try:
                    res_sp = least_squares(
                        fun=wrapped_F, jac=problem.J, x0=x0, method=scipy_method,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=100
                    )
                    hist_sp = [np.sum(problem.F(x) ** 2) for x in x_history]
                    
                    iters_sp = res_sp.niter if hasattr(res_sp, 'niter') else len(hist_sp) - 1
                    final_rss_sp = hist_sp[-1]
                except Exception:
                    hist_sp = [] 
                    iters_sp = np.nan
                    final_rss_sp = np.nan
                
                scipy_final_rss.append(final_rss_sp)
                scipy_iterations.append(iters_sp)
                
                rss_sec = np.clip(res_sec.rss_history, 1e-20, None)
                rss_sp = np.clip(hist_sp, 1e-20, None)
                
                line_sec, = ax.semilogy(rss_sec, color=method_palette.get("ModifiedGN"), linewidth=1.5)
                
                if len(hist_sp) > 0:
                    line_sp,  = ax.semilogy(rss_sp, color=method_palette.get("LM-lm"), linewidth=1.5)
                    if legend_handles is None:
                        legend_handles = [line_sec, line_sp]
                
                ax.text(0.95, 0.95, f"n: {n}\nm: {m}", color='black',
                        ha='right', va='top', transform=ax.transAxes, 
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))
                
                ax.grid(True, which="both", ls="-", alpha=0.2)
            
            ax.tick_params(axis='both', which='major')
            
            if i == 4:
                ax.set_xlabel("Iteration", fontsize=12)
            if j == 0:
                ax.set_ylabel("$\|F(x)\|^2$", fontsize=12)

    if legend_handles is not None:
        fig.legend(
            legend_handles, 
            ["Modified GN", "SciPy"],
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.95),
            ncol=2, 
            fontsize=12,
            frameon=True,
        )
    
    fig.suptitle(f"Convergence Matrix (5x5) for {problem_cls.__name__}", y=0.98, fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94]) 
    plt.show()

    print("\n" + "="*60)
    print(f" STATISTICAL SUMMARY FOR {problem_cls.__name__.upper()} ")
    print("="*60)
    
    if len(mgn_final_rss) > 0:
        print(f"{'Metric':<20} | {'Modified GN':<18} | {'SciPy':<18}")
        print("-"*60)
        
        mean_rss_mgn, std_rss_mgn = np.nanmean(mgn_final_rss), np.nanstd(mgn_final_rss)
        mean_rss_sp, std_rss_sp = np.nanmean(scipy_final_rss), np.nanstd(scipy_final_rss)
        
        print(f"{'Mean Final |F(x)|^2':<20} | {mean_rss_mgn:<18.4e} | {mean_rss_sp:<18.4e}")
        print(f"{'Std Final |F(x)|^2':<20} | {std_rss_mgn:<18.4e} | {std_rss_sp:<18.4e}")
        print("-"*60)
        
        mean_iter_mgn, std_iter_mgn = np.nanmean(mgn_iterations), np.nanstd(mgn_iterations)
        mean_iter_sp, std_iter_sp = np.nanmean(scipy_iterations), np.nanstd(scipy_iterations)
        
        print(f"{'Mean Iterations':<20} | {mean_iter_mgn:<18.2f} | {mean_iter_sp:<18.2f}")
        print(f"{'Std Iterations':<20} | {std_iter_mgn:<18.2f} | {std_iter_sp:<18.2f}")
    else:
        print("No valid configurations were evaluated to generate statistics.")
    print("="*60 + "\n")

def plot_optimizer_paths(
    problem_data,
    grid_size=200,
    padding=0.2
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
            cmap="viridis",
            alpha=0.4
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
            zorder=2
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
            zorder=2
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
            color="black",
            zorder=10
        )

        if problem.certified_solution is not None:

            x_star = problem.certified_solution

            ax.scatter(
                x_star[0],
                x_star[1],
                marker='*',
                s=280,
                label='Certified Solution',
                color='red',
                zorder=10,
                edgecolors='black'
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