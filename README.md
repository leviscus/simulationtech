# Poisson Problem Solver

This repository contains Python code for solving the Poisson-Dirichlet problem on a unit square mesh using the finite element method. The exact solution is \( 1 + x^2 + 2y^2 \).

## Instructions

1. **Set up the environment**:
   - Install Python and dependencies (e.g., `numpy`, `pytest`, `dolfinx`, `mpi4py`).

2. **Run the solver**:
   - Use the main script `poisson.py` to compute and save the solution:
     ```bash
     python poisson.py
     ```

3. **Run the tests**:
   - Verify correctness and convergence properties:
     ```bash
     pytest test_poisson_template.py
     ```

4. **Outputs**:
   - Solution files are saved in the `results/` directory in VTX format.

