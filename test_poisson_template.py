import pytest
import numpy as np
from ufl import le

from poisson import solve_poisson, errornorm

def test_exact_solution():
    """Test that P2 elements recover the quadratic solution exactly up to rounding error
    """
    uh, ue = solve_poisson(n=4, degree=2)
    error_H1 = errornorm(uh, ue, "H1")

    assert error_H1 < 1e-12
    

def test_convergence_P1():
    """Test that P1 elements converge linearly in the H1 norm and quadratically in the L2 norm
    """
    
    pass

# Test implementation below the placeholder
    mesh_sizes = [4, 8, 16, 32]  # List of mesh resolutions
    errors_L2 = []
    errors_H1 = []

    for n in mesh_sizes:
        uh, ue = solve_poisson(n=n, degree=1)  # Use P1 elements
        errors_L2.append(errornorm(uh, ue, "L2"))
        errors_H1.append(errornorm(uh, ue, "H1"))

    # Compute convergence rates
    rates_L2 = [np.log(errors_L2[i+1] / errors_L2[i]) / np.log(0.5) for i in range(len(errors_L2) - 1)]
    rates_H1 = [np.log(errors_H1[i+1] / errors_H1[i]) / np.log(0.5) for i in range(len(errors_H1) - 1)]

    # Assert expected convergence rates
    assert all(rate > 1.9 for rate in rates_L2)  # Expect quadratic convergence in L2 norm
    assert all(rate > 0.9 for rate in rates_H1)  # Expect linear convergence in H1 norm