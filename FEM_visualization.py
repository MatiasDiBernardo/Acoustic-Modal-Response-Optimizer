import pyvista as pv
import meshio

def mode_shape_visulization():
    # read your FEniCS‐compatible XDMF mesh
    m = meshio.read("room.xdmf")
    m.write("room_mesh.vtu")

    m_mode = meshio.read("mode1.xdmf")      # reads points, cells, and point_data
    m_mode.write("mode1.vtu")  

    geom = pv.read("room_mesh.vtu")
    m1   = pv.read("mode1.vtu")
    geom.point_data["mode1"] = m1.point_data[next(iter(m1.point_data))]  # whatever key meshio used

    p = pv.Plotter()
    p.add_mesh(geom, scalars="mode1", show_edges=True)
    p.add_scalar_bar("Mode 1 amplitude")
    p.show()
