import pyvista as pv
from matplotlib import cm
import meshio

name_geometry = "mallado/esfera_en_paralelepipedo_refined.msh"
name_pressure_field = "results2/pressure_evolution_p0_000030.vtu"

m = meshio.read(name_geometry)
m.write("room_mesh.vtu")

# Si el archivo no es vtu
# m_mode = meshio.read("mode1.xdmf")      # reads points, cells, and point_data
# m_mode.write("mode1.vtu")  

geom = pv.read("room_mesh.vtu")
m1   = pv.read(name_pressure_field)
geom.point_data["mode1"] = m1.point_data[next(iter(m1.point_data))]  # whatever key meshio used

p = pv.Plotter()
p.add_mesh(geom, scalars="mode1", opacity=0.5, show_edges=False)
p.add_volume(geom,scalars="mode1", opacity=0.5, shade=True, cmap=cm.get_cmap("coolwarm"))
p.add_scalar_bar("Air pressure")
p.show()
