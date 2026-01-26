# ev-MOGA

A Python implementation of the **ev-MOGA** (ε-dominance Multi-Objective Genetic Algorithm).

**Adolfo Hilario**  
ahilario@upv.es  
Systems & Control Engineering Department  
Universitat Politècnica de València

<br>

---

## Introduction

**ev-MOGA** is a multi-objective evolutionary algorithm originally developed by **Juan Manuel Herrero** and published on MathWorks (MATLAB Central):

- [ev-MOGA: Multiobjective Evolutionary Algorithm (MATLAB Central)](https://es.mathworks.com/matlabcentral/fileexchange/31080-ev-moga-multiobjective-evolutionary-algorithm)

It solves multi-objective optimization problems of the form:

$$
\min_{\mathbf{x}\in S}\ \mathbf{f}(\mathbf{x})
$$

where:

- $\mathbf{x} \in \mathbb{R}^m$ is the vector of **decision variables** (parameters),
- $\mathbf{f}: \mathbb{R}^m \rightarrow \mathbb{R}^n$ is the vector of **objective functions** to be minimized, and
- $S$ is the set of **feasible solutions** satisfying the decision-variable constraints.

Furthermore, ev-MOGA is an **elitist genetic algorithm** that approximates the efficient frontier in \(\mathbb{R}^n\) using the concept of **ε-dominance**.

<br>

---

## Quick start

Import the library:

```python
import evmoga as ev
import numpy as np
```

Build a configuration dictionary:

```python
eMOGA = {
    "objfun": objective_function,
    "iterationfun": iteration_function,
    "resultsfun": results_function,

    "objfun_dim": n_obj,
    "searchSpace_dim": n_var,
    "searchspaceUB": +np.pi * np.ones(n_var),
    "searchspaceLB": -np.pi * np.ones(n_var),

    "Nind_P": 500,
    "Generations": 200,
    "Nind_GA": 200,
    "n_div": [200 for _ in range(n_obj)],
}
```

Run the algorithm:

```python
eMOGA = ev.MOGA(eMOGA)
```
<br>

---

## Examples

### Basic examples

This repository includes two basic scripts:

- `evMOGA_example1.py`
- `evMOGA_example2.py`

### Portfolio selection examples

You can also find two portfolio-selection examples:

- `portfolio_selection_run_evMOGA_2obj`: Markowitz-style portfolio optimization (return vs. risk).
- `portfolio_selection_run_evMOGA_3obj`: incorporates **ESG** criteria alongside return and risk.

Real datasets are used. These datasets, located in the `Datasets/` folder, were kindly provided by **Francesco Cesarone**, who used them in:

> Cesarone, F., Martino, M. L., & Carleo, A. (2022). Does ESG impact really enhance portfolio profitability?. *Sustainability, 14*(4), 2050. https://doi.org/10.3390/su14042050
