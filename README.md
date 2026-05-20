# Optimization Project

Implementation of the **Modified Gauss-Newton method** together with a trust-region based solver for the associated \(L_2\)-norm subproblem.

The Modified Gauss-Newton scheme extends the classical Gauss-Newton method by introducing idea of quadratic regularization with the use of a sharp merit function.

The implementation follows the framework proposed in:

> Yu. Nesterov, *Modified Gauss–Newton scheme with worst case guarantees for global performance*, Optimization Methods and Software, 2007.

---


## Main Components

**`modified_gauss_newton.py`** - implementation of the Modified Gauss--Newton method.

**`newton_with_secular_equation.py`** - solver for the trust-region subproblem based on the secular equation and the algorithm described in *Trust Region Methods*.

**`problems/`** - collection of benchmark nonlinear systems, including:
- overdetermined problems,
- underdetermined problems.

**`notebooks/`** - numerical experiments, analysis, and visualization utilities.

---

## Installation

The project requires **Python 3.11** or newer.

Install dependencies with:

```bash
pip install -e .
```

---

## Benchmark Problems

The `problems/` directory contains 23 nonlinear systems used to evaluate the algorithms:

* **Overdetermined:** 13 problems sourced from the NIST Standard Reference Database [4]. These are systems with more equations than variables (m > n).
* **Underdetermined:** 10 problems sourced from Lukšan et al. [5]. These are systems where the number of equations is less than the number of variables (m < n).
  
---

## Experiments

The repository contains Jupyter notebooks for numerical experiments:

- `overdetermined_analysis.ipynb`
- `underdetermined_analysis.ipynb`

The notebooks compare the behaviour of the implemented methods on different classes of nonlinear systems and provide visualization of convergence properties.

---

## References

1. A. R. Conn, N. I. M. Gould, and Ph. L. Toint,  
   *Trust Region Methods*, SIAM, 2000.

2. Yu. Nesterov,  
   *Modified Gauss–Newton Scheme with Worst Case Guarantees for Global Performance*,  
   Optimization Methods and Software, 22(3), 469–483, 2007.

3. Yu. Nesterov and B. T. Polyak,  
   *Cubic Regularization of Newton Method and Its Global Performance*,  
   Technical Report, Université catholique de Louvain, 2003.

4. National Institute of Standards and Technology (NIST),  
   *Standard Reference Datasets for Nonlinear Regression*. [Website](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml)

5. L. Lukšan, C. Matonoha, and J. Vlček,  
   *Sparse Test Problems for Unconstrained Optimization*,  
   Technical Report V-1064, Prague, 2009. [PDF](https://www.cs.cas.cz/matonoha/download/V1064.pdf)

---

## Authors

- Katsiaryna Bokhan ([GitHub](https://github.com/kateqwerty001))
- Aleksandra Kwiatkowska ([GitHub](https://github.com/kwiatkowskaa))
