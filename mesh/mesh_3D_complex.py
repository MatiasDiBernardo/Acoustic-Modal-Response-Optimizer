import gmsh
import os
import sys

def create_complex_mesh(floor_points, height, source_pos, f_max, name, verbose= False):
    """Crea una malla para una sala con geometría arbitraria definida por una lista de puntos en el piso.

    Args:
        floor_points (list of tuple(x, y)): Lista de coordenadas que definen el contorno del piso en Z=0.
        height (float): Altura de la sala.
        source_pos (tuple(x, y, z)): Posición de la esfera fuente.
        f_max (float): Frecuencia máxima para resolución de malla.
        name (string): Nombre base del archivo de salida.
    """

    # Inicializar Gmsh
    gmsh.initialize()

    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add(name)

    # Crear directorio de salida si no existe
    output_directory = "mallado"
    os.makedirs(output_directory, exist_ok=True)

    # Parámetros de malla según frecuencia máxima
    c = 343.0
    lambda_max = c / f_max
    r_esfera = lambda_max / 20
    min_lc = r_esfera / 20
    max_lc = lambda_max / 10

    # Construir el contorno del piso con puntos y líneas
    point_tags = []
    for x, y in floor_points:
        pt_tag = gmsh.model.occ.addPoint(x, y, 0)
        point_tags.append(pt_tag)

    line_tags = []
    n_pts = len(point_tags)
    for i in range(n_pts):
        start = point_tags[i]
        end = point_tags[(i + 1) % n_pts]
        line_tag = gmsh.model.occ.addLine(start, end)
        line_tags.append(line_tag)

    # Crear loop de curvas y superficie de piso
    curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
    plane_surf = gmsh.model.occ.addPlaneSurface([curve_loop])

    # Extruir la superficie para formar el volumen de la sala
    extr = gmsh.model.occ.extrude(
        [(2, plane_surf)], 0, 0, height
    )
    # Extraer el tag del volumen (dim=3)
    room_volume = next(tag for dim, tag in extr if dim == 3)

    # Crear esfera fuente
    x_s, y_s, z_s = source_pos
    sphere_tag = gmsh.model.occ.addSphere(x_s, y_s, z_s, r_esfera)

    # Cortar la esfera del volumen de la sala
    domain, _ = gmsh.model.occ.cut(
        [(3, room_volume)], [(3, sphere_tag)]
    )
    gmsh.model.occ.synchronize()

    if not domain:
        print("ERROR: Corte sin volumen resultante.")
        gmsh.finalize()
        sys.exit(1)

    final_vol = domain[0][1]

    # Obtener superficies límite
    boundaries = gmsh.model.getBoundary([(3, final_vol)], oriented=False, recursive=False)
    surface_tags = [tag for dim, tag in boundaries if dim == 2]

    # Clasificar superficies por bounding box
    tags = {"inferior": [], "superior": [], "fuente": [], "paredes": []}
    eps = 1e-6
    for tag in surface_tags:
        bb = gmsh.model.getBoundingBox(2, tag)
        zmin, zmax = bb[2], bb[5]
        # Piso
        if abs(zmin) < eps and abs(zmax) < eps:
            tags["inferior"].append(tag)
        # Techo
        elif abs(zmin - height) < eps and abs(zmax - height) < eps:
            tags["superior"].append(tag)
        else:
            # Superficie de esfera
            dx = bb[3] - bb[0]
            dy = bb[4] - bb[1]
            dz = bb[5] - bb[2]
            diam = 2 * r_esfera
            tol = r_esfera * 0.5
            if abs(dx - diam) < tol and abs(dy - diam) < tol and abs(dz - diam) < tol:
                tags["fuente"].append(tag)
            else:
                tags["paredes"].append(tag)
    
    if verbose:
        print("Los tags de esta geometría son: ")
        print(tags)

    # Crear grupos físicos
    gmsh.model.addPhysicalGroup(3, [final_vol], name="dominio_volumenico", tag=100)
    gmsh.model.addPhysicalGroup(2, tags["inferior"], name="inferior", tag=1)
    gmsh.model.addPhysicalGroup(2, tags["superior"], name="superior", tag=2)
    gmsh.model.addPhysicalGroup(2, tags["fuente"], name="fuente", tag=7)
    gmsh.model.addPhysicalGroup(2, tags["paredes"], name="paredes", tag=5)

    # Configuración de malla
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_lc)
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Generar y escribir malla
    gmsh.model.mesh.generate(3)
    out_path = os.path.join(output_directory, f"{name}.msh")
    gmsh.write(out_path)

    if verbose:
        print(f"Malla guardada en: {out_path}")
        gmsh.fltk.run() 
    
    gmsh.finalize()
