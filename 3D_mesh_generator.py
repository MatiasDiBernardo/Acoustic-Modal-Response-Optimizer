import pygmsh
import meshio

# 1) Build your extruded geometry as before
floor_coords = [
    (0.0, 0.0),
    (3.0, 0.0),
    (3.0, 2.5),
    (1.5, 2.5),
    (1.5, 1.2),
    (0.0, 1.2),
]
Z = 2.2

with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(floor_coords, mesh_size=0.1)
    geom.extrude(poly, translation_axis=[0,0,Z], num_layers=10)
    msh = geom.generate_mesh()

# 2) Pull out _only_ the tetrahedral cells
tetra_cells = [c for c in msh.cells if c.type == "tetra"]

# 3) (Optionally) pull out cell_data for those tets, e.g. physical tags
tetra_data = {}
if "gmsh:physical" in msh.cell_data_dict:
    phys = msh.cell_data_dict["gmsh:physical"]
    tetra_phys = [d for (c,d) in zip(msh.cells, phys) if c.type=="tetra"]
    tetra_data["gmsh:physical"] = tetra_phys

# 4) Write a clean .xdmf containing only tets
clean = meshio.Mesh(
    points=msh.points,
    cells=tetra_cells,
    cell_data=tetra_data
)
meshio.write("room.xdmf", clean, file_format="xdmf")
