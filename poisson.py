from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
from ufl import SpatialCoordinate, TrialFunction, TestFunction, inner, grad, dx

def solve_poisson(n=8, degree=1):
    """Solves the Poisson-Dirichlet problem on the unit square with exact solution
    1 + x² + 2y²

    Args:
        n: Number of cells in x- and y-direction
        degree: Polynomial degree of the Lagrange finite elements

    Returns:
        uh: Numerical solution
        ue: Exact solution
    """
    
    # Create mesh and define function space
    msh = mesh.create_unit_square(
        comm=MPI.COMM_WORLD,
        nx=n,
        ny=n
    )

    V = fem.functionspace(
        mesh=msh,
        element=("P", degree)
    )

    # Define boundary condition
    tdim = msh.topology.dim # topological dimension of the mesh
    fdim = tdim - 1 # facet dimension
    msh.topology.create_connectivity(fdim, tdim) # what facets are connected to which cells
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(
        V=V,
        entity_dim=1,
        entities=boundary_facets
    )

    def manufactured_solution(x):
        """Defines a quadratic polynomial in x[0] and x[1]

        Args:
            x: Coordinates. x[0] = x-coordinate and x[1] = y-coordinate

        Returns:
            The function value of 1 + x² + 2y² at this point
        """
        return 1 + x[0]**2 + 2 * x[1]**2

    uD = fem.Function(V)
    uD.interpolate(manufactured_solution)
    

    bc = fem.dirichletbc(value=uD, dofs=boundary_dofs)

    # Define variational problem
    x = SpatialCoordinate(msh)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = fem.Constant(msh, -6.)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    # Compute solution
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    uh.name = "Solution u"

    # Exact solution
    ue = manufactured_solution(x)

    return uh, ue

def errornorm(uh, ue, norm="L2"):
    """Computes the L² norm or the H¹ norm of the error between the numerical and the exact solution

    Args:
        uh: Numerical solution
        ue: Exact solution
        norm: "L2" (default) or "H1"

    Returns:
        error: Value of the error
    """
    if norm=="H1":
        errorform = fem.form((uh - ue)**2 * dx + inner(grad(uh - ue), grad(uh - ue)) * dx)
    else:
        errorform = fem.form((uh - ue)**2 * dx)
    
    return np.sqrt(fem.assemble_scalar(errorform))

def save_solution(uh):
    """Exports the numerical solution in VTX format

    Args:
        uh: Numerical solution
    """
    msh = uh.function_space.mesh

    with io.VTXWriter(msh.comm, "results/poisson.bp", [uh]) as vtx:
        vtx.write(0.0)

if __name__ == "__main__":
    uh, ue = solve_poisson(
    n=4,
    degree=1
)

error_L2 = errornorm(uh, ue, "L2")
print("L2-error:", error_L2)
error_H1 = errornorm(uh, ue, "H1")
print("H1-error:", error_H1)

save_solution(uh)