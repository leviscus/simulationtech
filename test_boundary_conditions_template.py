import pytest
import numpy as np

import gmsh
from mpi4py import MPI
from dolfinx import fem, io, mesh
from ufl import Measure, dx
from boundary_conditions import mesh_benchmark_2d

def test_area_mesh_benchmark_2d():
    """Test that the domain has the expected area
    """

    domain, cell_tags, facet_tags = mesh_benchmark_2d()

    one = fem.Constant(domain, 1.)
    area = fem.assemble_scalar(fem.form(one * dx))

    assert area ==

def test_boundaries_mesh_benchmark_2d():
    """Test that the different parts of the boundary have the expected length
    """
    
    domain, cell_tags, facet_tags = mesh_benchmark_2d()

    one = fem.Constant(domain, 1.)
    ds = Measure("ds", domain=domain, subdomain_data=facet_tags)

    length_inlet = fem.assemble_scalar(fem.form(one * ds(1)))
    length_walls =
    length_obstacle =
    length_outlet =

    assert
