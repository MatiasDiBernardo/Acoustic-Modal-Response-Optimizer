import os # Importado para crear el directorio de salida
#import pygmsh
import gmsh
#import meshio 
import sys

def generate_mesh_parallelepiped(floor_coords, Z, source_position, f_max, 
                                 elements_per_wavelenght=10, name="room", output_path=None):
    """
    Genera una malla 3D de un paralelepípedo con una fuente esférica interna.

    Guarda un archivo .msh que puede ser cargado por solvers de Elementos Finitos.

    Args:
        floor_coords (tuple): Tupla (Lx, Ly) con las dimensiones de la base en metros.
        Z (float): Altura de la sala (coordenada Z) en metros.
        source_position (tuple): Tupla (X, Y, Z) con la posición del centro de la fuente en metros.
        f_max (int): Frecuencia máxima a representar para calcular el tamaño de malla.
        elements_per_wavelenght (int, optional): Nº de elementos que entran en la longitud de onda mínima. Defaults to 10.
        name (str, optional): Nombre base para el archivo de salida (sin extensión). Defaults to "room".
    """
    gmsh.initialize()
    gmsh.model.add("pulsating_sphere_room")
    if output_path is not None:
        output_directory = output_path
    else:
        # --- Creación de Directorio ---
        output_directory = "mallado"
        os.makedirs(output_directory, exist_ok=True)

    # --- Parámetros de Malla y Geometría ---
    c = 343  # Velocidad del sonido en m/s
    landa_f_max = c / f_max

    cte_f = elements_per_wavelenght
    cte_r = 12  # Elementos que definen el radio de la esfera

    Lx, Ly = floor_coords
    Lz = Z

    x_esfera, y_esfera, z_esfera = source_position
    r_esfera = landa_f_max / 10

    min_lc = r_esfera / cte_r
    max_lc = landa_f_max / cte_f

    # --- Verificación de Geometría ---
    dimensiones = [Lx, Ly, Lz]
    for i in range(len(source_position)):
        if source_position[i] + r_esfera >= dimensiones[i] or source_position[i] - r_esfera <= 0:
            print(f"ERROR: La esfera con centro en {source_position} y radio {r_esfera:.4f} está fuera del dominio [0,{Lx}]x[0,{Ly}]x[0,{Lz}].")
            gmsh.finalize()
            return

    # --- Creación de Geometría (OCC) ---
    esfera_tag_geom = gmsh.model.occ.addSphere(x_esfera, y_esfera, z_esfera, r_esfera)
    paralelepipedo_tag_geom = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

    try:
        dominio, _ = gmsh.model.occ.cut([(3, paralelepipedo_tag_geom)], [(3, esfera_tag_geom)])
    except Exception as e:
        print(f"ERROR: Falló la operación de corte (cut). ¿La esfera está completamente dentro de la caja?")
        print(f"Detalles: {e}")
        gmsh.finalize()
        sys.exit()

    gmsh.model.occ.synchronize()

    # --- Identificación y Etiquetado de Entidades ---
    if not dominio:
        print("ERROR: La operación de corte no produjo un volumen resultante.")
        gmsh.finalize()
        sys.exit()
        
    volume_tag = dominio[0][1]
    print(f"Se guardó el volumen como {volume_tag}")

    contornos = gmsh.model.getBoundary(dominio, combined=False, oriented=False, recursive=False)
    
    surface_tags = [tag for dim, tag in contornos if dim == 2]

    if len(surface_tags) != 7:
        print(f"ERROR: Se esperaban 7 superficies, pero se encontraron {len(surface_tags)}.")
        gmsh.finalize()
        sys.exit()
    else:
        print("INFO: Se encontraron las 7 superficies esperadas.")

    # --- Identificación de Superficies por Bounding Box ---
    tags_superficies_nombradas = { "superior": -1, "inferior": -1, "frontal": -1, "trasera": -1, "derecha": -1, "izquierda": -1, "fuente": -1 }
    epsilon = 1e-5

    for s_tag in surface_tags:
        bbox = gmsh.model.getBoundingBox(2, s_tag)
        # Identificar paredes planas
        if abs(bbox[2] - bbox[5]) < epsilon: # Plano Z cte
            if abs(bbox[2] - 0) < epsilon: tags_superficies_nombradas["inferior"] = s_tag
            elif abs(bbox[2] - Lz) < epsilon: tags_superficies_nombradas["superior"] = s_tag
        elif abs(bbox[0] - bbox[3]) < epsilon: # Plano X cte
            if abs(bbox[0] - 0) < epsilon: tags_superficies_nombradas["trasera"] = s_tag
            elif abs(bbox[0] - Lx) < epsilon: tags_superficies_nombradas["frontal"] = s_tag
        elif abs(bbox[1] - bbox[4]) < epsilon: # Plano Y cte
            if abs(bbox[1] - 0) < epsilon: tags_superficies_nombradas["izquierda"] = s_tag
            elif abs(bbox[1] - Ly) < epsilon: tags_superficies_nombradas["derecha"] = s_tag
        else: # Si no es plana en X, Y, o Z, debe ser la esfera
            tags_superficies_nombradas["fuente"] = s_tag

    print("\n--- Tags de Superficies Identificadas (Resultado Final) ---")
    for nombre, tag_id in tags_superficies_nombradas.items():
        print(f"Superficie '{nombre}': tag geométrico {tag_id}")

    # --- Asignación de Grupos Físicos ---
    # *** CAMBIO: Definir tags físicos en un diccionario para mayor claridad ***
    physical_tags = {
        "superior": 1, "inferior": 2, "frontal": 3, "trasera": 4, 
        "derecha": 5, "izquierda": 6, "fuente": 7, "dominio_volumetrico": 100
    }
    
    print("\n--- Asignando Grupos Físicos ---")
    for nombre_superficie, geom_tag in tags_superficies_nombradas.items():
        if geom_tag == -1:
            print(f"ERROR CRÍTICO: No se identificó la superficie '{nombre_superficie}'. Abortando.")
            gmsh.finalize()
            sys.exit()
        phys_tag = physical_tags[nombre_superficie]
        gmsh.model.addPhysicalGroup(2, [geom_tag], tag=phys_tag)
        gmsh.model.setPhysicalName(2, phys_tag, nombre_superficie)
        print(f"Grupo Físico '{nombre_superficie}' (Tag: {phys_tag}) asignado a la geometría {geom_tag}.")

    vol_phys_tag = physical_tags["dominio_volumetrico"]
    gmsh.model.addPhysicalGroup(3, [volume_tag], tag=vol_phys_tag)
    gmsh.model.setPhysicalName(3, vol_phys_tag, "dominio_volumetrico")
    print(f"Grupo Físico 'dominio_volumetrico' (Tag: {vol_phys_tag}) asignado a la geometría {volume_tag}.")

    # --- Configuración y Generación de Malla ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Frontal-Delaunay (Netgen)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_lc)
    print(f"\nTamaño característico de malla: Min={min_lc:.4f}, Max={max_lc:.4f}")

    gmsh.model.mesh.setOrder(2)
    print("Establecido orden de la malla a 2 (elementos cuadráticos).")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    print("Establecida versión del archivo MSH a 4.1")

    print("Generando malla 3D de segundo orden...")
    gmsh.model.mesh.generate(3)
    print("Malla generada.")
    
    # *** CAMBIO: Asegurarse de que el nombre del archivo tenga la extensión .msh ***
    # Eliminar posible extensión previa y añadir la correcta
    nombre_base, _ = os.path.splitext(name)
    mesh_output_filename = os.path.join(output_directory, f"{nombre_base}.msh")
    
    print(f"Guardando malla en: {mesh_output_filename}")
    gmsh.write(mesh_output_filename)
    print("Malla guardada.")

    # Descomenta la siguiente línea si quieres visualizar la malla de forma interactiva
    # gmsh.fltk.run() 

    gmsh.finalize()
    print("Gmsh finalizado.")