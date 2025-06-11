import gmsh
import os
import sys

def create_complex_mesh(floor_points, height, source_pos, f_max, name):
    """Crea una malla para una sala con geometría arbitraria definidapor una lista de puntos en el piso.

    Args:
        floor_points (list of tuple(x, y)): Lista de coordenadas que definen el contorno del piso en Z=0.
        height (float): Altura de la sala.
        source_pos (tuple(x, y, z)): Posición de la esfera fuente.
        f_max (float): Frecuencia máxima para resolución de malla.
        name (string): Nombre base del archivo de salida.
    """
    gmsh.initialize()
    gmsh.model.add(name)

    # Directorio de salida
    output_directory = "mallado"
    os.makedirs(output_directory, exist_ok=True)

    # Parámetros de malla según f_max
    c = 343
    lambda_max = c / f_max
    min_lc = 0.1 / 10   # radio esfera fijo 0.1 m
    max_lc = lambda_max / 10

    # Crear contorno de piso en Z=0
    # Generamos polígonocerrado con addPolygon (X1, Y1, Z1, X2, Y2, Z2, ...)
    polygon_coords = []
    for x, y in floor_points:
        polygon_coords.extend([x, y, 0])
    curve_loop = gmsh.model.occ.addPolygon(*polygon_coords)
    plane_surface = gmsh.model.occ.addPlaneSurface([curve_loop])

    # Extruir la superficie del piso a la altura deseada
    extrusions = gmsh.model.occ.extrude(
        [(2, plane_surface)], 0, 0, height,
        numElements=[1], # opcional: un bloque vertical
    )
    # Obtener el volumen de la sala (dim=3)
    room_volume = next(tag for dim, tag in extrusions if dim == 3)

    # Crear esfera fuente
    x_s, y_s, z_s = source_pos
    sphere_tag = gmsh.model.occ.addSphere(x_s, y_s, z_s, 0.1)

    # Cortar esfera del volumen de la sala
    domaine, _ = gmsh.model.occ.cut(
        [(3, room_volume)], [(3, sphere_tag)]
    )
    gmsh.model.occ.synchronize()

    # Extraer tag de volumen resultante
    if not domaine:
        print("ERROR: Corte sin volumen resultante.")
        gmsh.finalize(); sys.exit(1)
    final_vol = domaine[0][1]

    # Obtener todas las superficies límite
    boundaries = gmsh.model.getBoundary([(3, final_vol)], oriented=False,
                                        recursive=False)
    surface_tags = [tag for dim, tag in boundaries if dim == 2]

    # Clasificar superficies según bounding box
    tags = {"inferior": [], "superior": [], "fuente": [], "paredes": []}
    eps = 1e-6
    for tag in surface_tags:
        bb = gmsh.model.getBoundingBox(2, tag)
        zmin, zmax = bb[2], bb[5]
        # Piso (Z=0)
        if abs(zmin) < eps and abs(zmax) < eps:
            tags["inferior"].append(tag)
        # Techo (Z=height)
        elif abs(zmin - height) < eps and abs(zmax - height) < eps:
            tags["superior"].append(tag)
        else:
            # ¿Es esfera?
            dx = bb[3] - bb[0]
            dy = bb[4] - bb[1]
            dz = bb[5] - bb[2]
            if abs(dx - 0.2) < 0.05 and abs(dy - 0.2) < 0.05 and abs(dz - 0.2) < 0.05:
                tags["fuente"].append(tag)
            else:
                tags["paredes"].append(tag)

    # Crear grupos físicos
    # Volumen
    gmsh.model.addPhysicalGroup(3, [final_vol], name="dominio_volumenico", tag=100)
    # Piso y techo y fuente
    gmsh.model.addPhysicalGroup(2, tags["inferior"], name="inferior", tag=1)
    gmsh.model.addPhysicalGroup(2, tags["superior"], name="superior", tag=2)
    gmsh.model.addPhysicalGroup(2, tags["fuente"], name="fuente", tag=7)
    # Paredes (todas)
    gmsh.model.addPhysicalGroup(2, tags["paredes"], name="paredes", tag=5)

    # Opciones de malla
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_lc)
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Generar malla y escribir
    gmsh.model.mesh.generate(3)
    out = os.path.join(output_directory, f"{name}.msh")
    gmsh.write(out)
    print(f"Malla guardada en: {out}")

    # Descomenta si quieres ver la malla antes de salir
    gmsh.fltk.run() 

    gmsh.finalize()
