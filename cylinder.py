import gmsh
gmsh.initialize()
gmsh.model.add("DFG Benchmark Geometry")

# Parameters
channel_length = 2.2  # Length of the channel
channel_height = 0.41  # Height of the channel
cylinder_center = (0.2, 0.2)  # Center of the cylinder
cylinder_radius = 0.05  # Radius of the cylinder
mesh_size = 0.02  # Global mesh size

# Create the channel (rectangle)
rectangle = gmsh.model.occ.addRectangle(0, 0, 0, 2.2, 0.41)

# Create the cylinder (disk)
cylinder = gmsh.model.occ.addDisk(0.2, 0.2, 0, 0.05, 0.05)

gdim = 2 # geometric dimension of this model

# Subtract the cylinder from the channel
channel_with_hole = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, cylinder)],removeTool=False)

# Synchronize the CAD representation
gmsh.model.occ.synchronize()

surface_entities = [entity[1] for entity in gmsh.model.getEntities(dim=gdim)]
gmsh.model.addPhysicalGroup(gdim, surface_entities, tag=1)


# Mesh the domain
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

gmsh.model.mesh.generate(gdim)

# Save the mesh to a file for FEniCSx
gmsh.write("dfg_benchmark.msh")

from mpi4py import MPI
import numpy as np
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import SpatialCoordinate, conditional, And, gt, lt, TrialFunction, TestFunction, inner, grad, dx, ds
from pathlib import Path

gmsh.write("dfg_benchmark.msh")  # Ensure the mesh is written in .msh format
gmsh.finalize()  # Finalize Gmsh before reading into FEniCSx

# Load the mesh into FEniCSx
domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
    "dfg_benchmark.msh",
    MPI.COMM_WORLD,
    gdim=gdim
)