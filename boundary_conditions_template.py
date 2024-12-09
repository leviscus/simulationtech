import gmsh
from mpi4py import MPI
import numpy as np
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import Measure, FacetNormal, SpatialCoordinate, conditional, And, gt, lt, TrialFunction, TestFunction, inner, grad, dx
from pathlib import Path

def mesh_room():
    """Creates a mesh for a room with two straight walls and two curved walls.

    Returns:
        domain, cell_tags, facet_tags: A mesh with tags, as returned by dolfinx.io.gmshio.model_to_mesh
    """

    gmsh.initialize()

    disk = gmsh.model.occ.addDisk(0, 0, 0, 10, 10)
    rectangle1 = gmsh.model.occ.addRectangle(-10, 5, 0, 20, 5)
    rectangle2 = gmsh.model.occ.addRectangle(-10, -10, 0, 20, 5)

    gdim = 2 # geometric dimension of this model

    gmsh.model.occ.cut([(gdim, disk)], [(gdim, rectangle1), (gdim, rectangle2)])

    gmsh.model.occ.synchronize()
    domain_entities = [entity[1] for entity in gmsh.model.getEntities(dim=gdim)]
    gmsh.model.addPhysicalGroup(gdim, domain_entities, tag=1)

    boundary_entities = [entity[1] for entity in gmsh.model.getEntities(dim=gdim-1)]
    gmsh.model.addPhysicalGroup(gdim-1, boundary_entities, tag=1)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    gmsh.write("room.msh")

    domain, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        model=gmsh.model,
        comm=MPI.COMM_WORLD,
        rank=0,
        gdim=2
    )

    gmsh.clear()
    gmsh.finalize()

    return domain, cell_tags, facet_tags


def mesh_benchmark_2d():
    """Creates a mesh for the two-dimensional DFG benchmark for flow around a cyclinder.

    The inlet is marked with facet tag 1.
    The top and bottom walls are marked with facet tag 2.
    The obstacle boundary is marked with facet tag 3.
    The outlet is marked with facet tag 4.

    Returns:
        domain, cell_tags, facet_tags: A mesh with tags, as returned by dolfinx.io.gmshio.model_to_mesh
    """

    gmsh.initialize()

    channel_length = 2.2
    channel_height = .41
    obstacle_centre = [.2, .2]
    obstacle_radius = .05

    channel = gmsh.model.occ.addRectangle(
        x=0,
        y=0,
        z=0,
        dx=channel_length,
        dy=channel_height
    )
    obstacle = gmsh.model.occ.addDisk(
        xc=obstacle_centre[0],
        yc=obstacle_centre[1],
        zc=0,
        rx=obstacle_radius,
        ry=obstacle_radius
    )

    gdim = 2 # geometric dimension of this model

    gmsh.model.occ.cut([(gdim, channel)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

    # Domain
    domain_entities = [entity[1] for entity in gmsh.model.getEntities(dim=gdim)]
    gmsh.model.addPhysicalGroup(gdim, domain_entities, tag=1)

    # Pieces of the boundary
    boundary_entities = [entity[1] for entity in gmsh.model.getEntities(dim=gdim-1)]

    inlet_entities = []
    wall_entities = []
    obstacle_entities = []
    outlet_entities = []

    for boundary_part in boundary_entities:
        xc = gmsh.model.occ.getCenterOfMass(
            dim=gdim-1,
            tag=boundary_part
        )

        if np.isclose(xc[0], 0.):
            # This piece of the boundary is part of the inlet
            inlet_entities.append(boundary_part)
        elif np.isclose(xc[0], channel_length):
            # This piece of the boundary is part of the outlet
            outlet_entities.append(boundary_part)
        elif np.isclose(xc[1], 0.) or np.isclose(xc[1], channel_height):
            # This piece of the boundary is part of the bottom or top wall
            wall_entities.append(boundary_part)
        else:
            # This piece of the boundary is part of the obstacle
            obstacle_entities.append(boundary_part)

    # We label the inlet with tag 1
    gmsh.model.addPhysicalGroup(gdim-1, inlet_entities, tag=1)
    gmsh.model.setPhysicalName(gdim-1, 1, "Inlet")

    # We label the top and bottom walls with tag 2
    gmsh.model.addPhysicalGroup(gdim-1, wall_entities, tag=2)
    gmsh.model.setPhysicalName(gdim-1, 2, "Walls")

    # We label the obstacle with tag 3
    gmsh.model.addPhysicalGroup(gdim-1, obstacle_entities, tag=3)
    gmsh.model.setPhysicalName(gdim-1, 3, "Obstacle")

    # We label the outlet with tag 4
    gmsh.model.addPhysicalGroup(gdim-1, outlet_entities, tag=4)
    gmsh.model.setPhysicalName(gdim-1, 4, "Outlet")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", obstacle_radius/40)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", obstacle_radius/40)
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    gmsh.write("benchmark_2d.msh")

    domain, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        model=gmsh.model,
        comm=MPI.COMM_WORLD,
        rank=0,
        gdim=2
    )

    gmsh.clear()
    gmsh.finalize()

    return domain, cell_tags, facet_tags


def solve_temperature(mesh):
    domain, cell_tags, facet_tags = mesh()
    
    # Add information about boundary sections to the ds measure
    ds = Measure("ds", domain=domain, subdomain_data=facet_tags)

    V = fem.functionspace(domain, ("P", 1))

    k = fem.Constant(domain, 2.4) # heat conductivity
    alpha = fem.Constant(domain, 5.) # heat transfer coefficient in the Robin boundary condition
    T3 = 35. # temperature of the circular part of the boundary
    T4 = fem.Constant(domain, 10.) # ambient temperature outside the outlet
    f = fem.Constant(domain, 0.) # source term

    # Dirichlet boundary condition on Section #3
    boundary_dofs = fem.locate_dofs_topological(
        V=V,
        entity_dim=1,
        entities=facet_tags.find(3)
    )
    bc = fem.dirichletbc(T3, boundary_dofs, V)

    # Variational form with
    # homogeneous Neumann bc's on Sections #1 & #2
    # Robin bc's on Section #4
    T = TrialFunction(V)
    v = TestFunction(V)
    a = k * inner(grad(T), grad(v)) * dx + alpha *  T * v * ds(4)
    L = f * v * dx + alpha * T4 * v * ds(4)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "richardson", "pc_type": "lu",  "pc_hypre_type": "boomeramg"})

    t_start = MPI.Wtime()
    Th = problem.solve()
    t_finish = MPI.Wtime()

    print("Linear system of equations solved in" , t_finish-t_start, "seconds")

    Th.name = "Temperature"

    n = FacetNormal(domain)
    print('Net heat influx:', fem.assemble_scalar(fem.form(k*inner(grad(Th),n)*ds)))

    return Th


def save_solution(uh):
    """Exports the numerical solution in VTX format

    Args:
        uh: Numerical solution
    """
    msh = uh.function_space.mesh

    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True, parents=True)

    # We replace any whitespace characters in the name of uh with an underscore
    filename = "_".join(uh.name.split())

    with io.VTXWriter(msh.comm, results_folder / (filename + ".bp"), [uh]) as vtx:
        vtx.write(0.0)


if __name__ == "__main__": 
    Th = solve_temperature(mesh_benchmark_2d)
    save_solution(Th)
