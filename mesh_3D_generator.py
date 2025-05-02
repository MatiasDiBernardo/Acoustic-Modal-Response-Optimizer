import pygmsh
import meshio

def generate_mesh(floor_coords, Z):
    """Generate the mesh in a room.xdmf file that can be loaded by FEniCS

    Args:
        floor_coords (list): List of points (X, Y) with the vertices of the geometry
        Z (int): Hight of the room (Z coordenate) in meters
    """
    # Definir esto en base a la frecuencia para optimizar
    mesh_setp_size = 0.2  # Original es 0.1
    layers_in_extrapolate = 8  # Original 10

    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(floor_coords, mesh_size=mesh_setp_size)
        geom.extrude(poly, translation_axis=[0,0,Z], num_layers=layers_in_extrapolate)
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
