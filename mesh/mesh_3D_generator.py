import os # Importado para crear el directorio de salida
import pygmsh
import gmsh
import meshio 
import sys

def generate_mesh_parallelepiped(floor_coords, Z, source_position, f_max):
    """Generate the mesh in a room.msh file that can be loaded by FEniCS

    Args:
        floor_coords (list): List of points (X, Y) with the vertices of the geometry
        Z (int): Hight of the room (Z coordenate) in meters
        source_position (tuple[X, Y, Z]): Position of the source in space in meters
        f_max (integer): Maximum frecuency to representate
    """
    gmsh.initialize()
    gmsh.model.add("pulsating_sphere_very_small_refined") # Nombre actualizado

    # Crear directorio de salida si no existe
    output_directory = "mallado"
    os.makedirs(output_directory, exist_ok=True)

    #Se calcula la longitud de onda mas cortrta a representar 
    c = 343
    landa_max = c/f_max

    #Constante de superacion: Bajar si toma demasiado tiempo, subir si falta definicion
    cte_f = 10 #Cuantos elememtos definimos que entren en landa mas chico Lanxa_max
    cte_r = 1 #Cuantos elementos definimos que entren en el radio de la esfera

    # Se definen las dimensiones del paralelepípedo y la ubicación de la esfera (todo en S.I)
    Lx, Ly = floor_coords
    Lz = Z

    # Fuente esférica
    x_esfera, y_esfera , z_esfera = source_position 
    r_esfera = landa_max/cte_f # Radio de la esfera interior

    #Se definen las dimensiones maximas y minimas de los elementos
    min_lc = r_esfera / cte_r
    max_lc = landa_max / cte_f # 


    dimensiones = [Lx, Ly, Lz]
    #Verificamos que la esfera no intercecte con las superficies de borde
    for i in range(len(source_position)):
        if source_position[i] + r_esfera >= dimensiones[i] or source_position[i] - r_esfera < 0:
            print("ERROR: La esfera esta fuera de dominio, redefina la ubicacion")
            return 

    # Primero se busca generar un sólido que representa el medio, el dominio desde el punto de vista matemático
    # Creamos la esfera
    esfera_tag_geom = gmsh.model.occ.addSphere(x_esfera, y_esfera, z_esfera, r_esfera) # Guardamos el tag de la esfera

    # Creamos el paralelepípedo
    paralelepipedo_tag_geom = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz) # Guardamos el tag del paralelepípedo

    # Se sustrae la esfera de la caja
    try:
        dominio, _ = gmsh.model.occ.cut([(3, paralelepipedo_tag_geom)], [(3, esfera_tag_geom)])
    except Exception as e:
        print(f"ERROR: Falló la operación de corte (cut). ¿La esfera está completamente dentro de la caja?")
        print(f"Detalles: {e}")
        gmsh.finalize()
        sys.exit()
        
    gmsh.model.occ.synchronize() # Hasta ahora tenemos un volumen, y las superficies de las paredes y esfera

    # --- Ahora hay que etiquetar los grupos físicos: Volumen, y superficies ---
    if not dominio:
        print("ERROR: La operación de corte no produjo un volumen resultante (dominio vacío).")
        gmsh.finalize()
        sys.exit()
        
    volume_tag = dominio[0][1] # Extrae el tag numérico del volumen
    print(f"Se guardó el volumen como {volume_tag}")

    # Obtener los límites (superficies) del volumen resultante
    contornos = gmsh.model.getBoundary(dominio, combined=False, oriented=False, recursive=False)
    print(f"Límites encontrados para el volumen {volume_tag}: {contornos}")


    # Inicializamos las paredes asumiendo que:
    # X+ -> dirección frontal
    # Y+ -> dirección derecha
    # Z+ -> dirección vertical (superior)

    tags_superficies_nombradas = {
        "superior": -1,
        "inferior": -1,
        "frontal": -1,
        "trasera": -1,
        "derecha": -1,
        "izquierda": -1,
        "fuente": -1
    }

    surface_tags = [] # lista contenedora de tags de superficie

    # Esto recorre toda la tupla, guardando todos los tags en una lista q sean superficie
    for dim, tag in contornos:
        if dim == 2:
            surface_tags.append(tag)

    if len(surface_tags) != 7: # verifica que solo hagan 7 superficies, 6 de las paredes, una de la esfera
        print(f"ERROR: Se esperaban 7 superficies límite (2D), pero se encontraron {len(surface_tags)}.")
        if len(surface_tags) == 0:
            print("ERROR FATAL: No se encontraron superficies límite.")
            gmsh.finalize()
            sys.exit()
    else:
        print("INFO: Se encontraron las 7 superficies esperadas.")


    # Obtener bounding box para todas las superficies encontradas
    bbox_list = [None] * len(surface_tags)
    for i_idx in range(len(surface_tags)): 
        try:
            bbox_list[i_idx] = gmsh.model.getBoundingBox(2, surface_tags[i_idx])
        except Exception as e_bbox:
            print(f"ERROR obteniendo BBox para tag {surface_tags[i_idx]}: {e_bbox}")
            bbox_list[i_idx] = [0]*6 # Poner un valor por defecto para evitar errores posteriores


    # Ya tenemos las bounding boxes, ahora identificamos cada superficie
    epsilon = 1e-5 # Tolerancia para la planitud de las caras
    print("\n--- Identificación de Superficies ---")
    for i_idx in range(len(bbox_list)): 
        current_surface_tag = surface_tags[i_idx]
        current_bb = bbox_list[i_idx] 
        
        identity_found_for_surface_i = False 

        # --- Comprobación de Paredes ---
        if abs(current_bb[2] - current_bb[5]) < epsilon: 
            if abs(current_bb[2] - 0) < epsilon: 
                tags_superficies_nombradas["inferior"] = current_surface_tag
                identity_found_for_surface_i = True
            elif abs(current_bb[2] - Lz) < epsilon: 
                tags_superficies_nombradas["superior"] = current_surface_tag
                identity_found_for_surface_i = True
        
        if not identity_found_for_surface_i and abs(current_bb[0] - current_bb[3]) < epsilon: 
            if abs(current_bb[0] - 0) < epsilon: 
                tags_superficies_nombradas["trasera"] = current_surface_tag
                identity_found_for_surface_i = True
            elif abs(current_bb[0] - Lx) < epsilon: 
                tags_superficies_nombradas["frontal"] = current_surface_tag
                identity_found_for_surface_i = True

        if not identity_found_for_surface_i and abs(current_bb[1] - current_bb[4]) < epsilon: 
            if abs(current_bb[1] - 0) < epsilon: 
                tags_superficies_nombradas["izquierda"] = current_surface_tag
                identity_found_for_surface_i = True
            elif abs(current_bb[1] - Ly) < epsilon: 
                tags_superficies_nombradas["derecha"] = current_surface_tag
                identity_found_for_surface_i = True
        
        # --- Comprobación de Fuente ---
        if not identity_found_for_surface_i:
            is_flat_X = abs(current_bb[0] - current_bb[3]) < epsilon
            is_flat_Y = abs(current_bb[1] - current_bb[4]) < epsilon
            is_flat_Z = abs(current_bb[2] - current_bb[5]) < epsilon

            dim_x_bb = current_bb[3] - current_bb[0]
            dim_y_bb = current_bb[4] - current_bb[1]
            dim_z_bb = current_bb[5] - current_bb[2]
            expected_diameter = 2 * r_esfera
            diameter_epsilon = r_esfera * 0.5 

            is_sphere_dims = (abs(dim_x_bb - expected_diameter) < diameter_epsilon and
                            abs(dim_y_bb - expected_diameter) < diameter_epsilon and
                            abs(dim_z_bb - expected_diameter) < diameter_epsilon)

            if not is_flat_X and not is_flat_Y and not is_flat_Z and is_sphere_dims:
                if tags_superficies_nombradas["fuente"] == -1:
                    tags_superficies_nombradas["fuente"] = current_surface_tag
                else:
                    print(f"  ADVERTENCIA: Tag {current_surface_tag} parece ser otra fuente, pero 'fuente' ya estaba asignada a {tags_superficies_nombradas['fuente']}.")


    print("\n--- Tags de Superficies Identificadas (Resultado Final) ---")
    for nombre, tag_id in tags_superficies_nombradas.items():
        if tag_id != -1:
            print(f"Superficie '{nombre}': tag geométrico {tag_id}")
        else:
            print(f"Superficie '{nombre}': NO IDENTIFICADA (tag geométrico {tag_id})")

    # --- Asignar Grupos Físicos ---
    physical_tags_map = {
        "superior": 1,
        "inferior": 2,
        "frontal": 3,
        "trasera": 4,
        "derecha": 5,
        "izquierda": 6,
        "fuente": 7
    }
    volumen_phys_tag = 100

    print("\n--- Asignando Grupos Físicos ---")
    missing_surfaces = False
    for nombre_superficie, geom_tag_superficie in tags_superficies_nombradas.items():
        if geom_tag_superficie != -1:
            if nombre_superficie in physical_tags_map:
                phys_tag_para_esta_superficie = physical_tags_map[nombre_superficie]
                gmsh.model.addPhysicalGroup(2, [geom_tag_superficie], name=nombre_superficie, tag=phys_tag_para_esta_superficie)
                print(f"Grupo Físico de Superficie '{nombre_superficie}' (Tag Físico: {phys_tag_para_esta_superficie}) asignado a la entidad geométrica {geom_tag_superficie}.")
            else:
                print(f"ADVERTENCIA: La superficie '{nombre_superficie}' ({geom_tag_superficie}) fue identificada pero no tiene un tag físico predefinido en 'physical_tags_map'.")
        else:
            print(f"ERROR CRÍTICO: No se pudo identificar la superficie geométrica para '{nombre_superficie}', no se creará Grupo Físico.")
            missing_surfaces = True

    if missing_surfaces:
        print("ERROR FATAL: Faltan identificaciones de superficies geométricas. Abortando antes de mallar.")
        gmsh.finalize()
        sys.exit()


    gmsh.model.addPhysicalGroup(3, [volume_tag], name="dominio_volumetrico", tag=volumen_phys_tag)
    print(f"Grupo Físico de Volumen 'dominio_volumetrico' (Tag Físico: {volumen_phys_tag}) asignado a la entidad geométrica {volume_tag}.")


    # --- Configuración y Generación de Malla ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Netgen



    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_lc)
    print(f"Tamaño característico de malla: Min={min_lc}, Max={max_lc}")

    # Mantenemos orden 2 para la geometría de la esfera
    gmsh.model.mesh.setOrder(2)
    print("Establecido orden de la malla a 2 (elementos cuadráticos).")

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    print("Establecida versión del archivo MSH a 4.1")

    print("Generando malla 3D de segundo orden...")
    gmsh.model.mesh.generate(3)
    print("Malla generada.")

    # *** CAMBIO: Nombre de archivo para malla refinada ***
    mesh_output_filename = os.path.join(output_directory, "room.msh")
    print(f"Guardando malla en: {mesh_output_filename}")
    gmsh.write(mesh_output_filename)
    print("Malla guardada.")

    # Descomenta si quieres ver la malla antes de salir
    #gmsh.fltk.run() 

    gmsh.finalize()
    print("Gmsh finalizado.")

def generate_mesh_for_modal(floor_coords, Z):
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

#Tarea pendiente    
'''
    def generate_mesh_complex(floor_coords, Z, source_position, f_max):
    """Generate the mesh in a room.msh file that can be loaded by FEniCS

    Args:
        floor_coords (list): List of points (X, Y) with the vertices of the geometry
        Z (int): Hight of the room (Z coordenate) in meters
        source_position (tuple[X, Y, Z]): Position of the source in space in meters
        f_max (integer): Maximum frecuency to representate
    """
    gmsh.initialize()
    gmsh.model.add("pulsating_sphere_very_small_refined") # Nombre actualizado


    # Crear directorio de salida si no existe
    output_directory = "mallado"
    os.makedirs(output_directory, exist_ok=True)

    #Se calcula la longitud de onda mas cortrta a representar 
    c = 343
    landa_max = c/f_max

    #Constante de superacion: Bajar si toma demasiado tiempo, subir si falta definicion
    cte_f = 10 #Cuantos elememtos definimos que entren en landa mas chico Lanxa_max
    cte_r = 6 #Cuantos elementos definimos que entren en el radio de la esfera

    # Se definen las dimensiones del paralelepípedo y la ubicación de la esfera (todo en S.I)
    Lx, Ly = floor_coords
    Lz = Z

    # Fuente esférica
    x_esfera, y_esfera , z_esfera = source_position 
    r_esfera = landa_max/cte_f # Radio de la esfera interior

    #Se definen las dimensiones maximas y minimas de los elementos
    min_lc = r_esfera / cte_r
    max_lc = landa_max / cte_f # 


    print(f"Dominio: Lx={Lx}, Ly={Ly}, Lz={Lz}")
    print(f"Esfera: Centro=({x_esfera},{y_esfera},{z_esfera}), Radio={r_esfera}")
    print(f"Esfera: Centro=({x_esfera},{y_esfera},{z_esfera}), Radio={r_esfera}")


    dimensiones = [Lx, Ly, Lz]
    #Verificamos que la esfera no intercecte con las superficies de borde
    for i in range(len(source_position)):
        if source_position[i] + r_esfera >= dimensiones[i] or source_position[i] - r_esfera < 0:
            print("ERROR: La esfera esta fuera de dominio, redefina la ubicacion")
            return 


    # Primero se busca generar un sólido que representa el medio, el dominio desde el punto de vista matemático
    # Creamos la esfera
    esfera_tag_geom = gmsh.model.occ.addSphere(x_esfera, y_esfera, z_esfera, r_esfera) # Guardamos el tag de la esfera

    # Creamos el paralelepípedo
    paralelepipedo_tag_geom = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz) # Guardamos el tag del paralelepípedo

    # Se sustrae la esfera de la caja
    try:
        dominio, _ = gmsh.model.occ.cut([(3, paralelepipedo_tag_geom)], [(3, esfera_tag_geom)])
    except Exception as e:
        print(f"ERROR: Falló la operación de corte (cut). ¿La esfera está completamente dentro de la caja?")
        print(f"Detalles: {e}")
        gmsh.finalize()
        sys.exit()
        
    gmsh.model.occ.synchronize() # Hasta ahora tenemos un volumen, y las superficies de las paredes y esfera


    # --- Ahora hay que etiquetar los grupos físicos: Volumen, y superficies ---
    if not dominio:
        print("ERROR: La operación de corte no produjo un volumen resultante (dominio vacío).")
        gmsh.finalize()
        sys.exit()
        
    volume_tag = dominio[0][1] # Extrae el tag numérico del volumen
    print(f"Se guardó el volumen como {volume_tag}")

    # Obtener los límites (superficies) del volumen resultante
    contornos = gmsh.model.getBoundary(dominio, combined=False, oriented=False, recursive=False)
    print(f"Límites encontrados para el volumen {volume_tag}: {contornos}")


    # Inicializamos las paredes asumiendo que:
    # X+ -> dirección frontal
    # Y+ -> dirección derecha
    # Z+ -> dirección vertical (superior)

    tags_superficies_nombradas = {
        "superior": -1,
        "inferior": -1,
        "frontal": -1,
        "trasera": -1,
        "derecha": -1,
        "izquierda": -1,
        "fuente": -1
    }

    surface_tags = [] # lista contenedora de tags de superficie

    # Esto recorre toda la tupla, guardando todos los tags en una lista q sean superficie
    for dim, tag in contornos:
        if dim == 2:
            surface_tags.append(tag)

    if len(surface_tags) != 7: # verifica que solo hagan 7 superficies, 6 de las paredes, una de la esfera
        print(f"ERROR: Se esperaban 7 superficies límite (2D), pero se encontraron {len(surface_tags)}.")
        if len(surface_tags) == 0:
            print("ERROR FATAL: No se encontraron superficies límite.")
            gmsh.finalize()
            sys.exit()
    else:
        print("INFO: Se encontraron las 7 superficies esperadas.")


    # Obtener bounding box para todas las superficies encontradas
    bbox_list = [None] * len(surface_tags)
    for i_idx in range(len(surface_tags)): 
        try:
            bbox_list[i_idx] = gmsh.model.getBoundingBox(2, surface_tags[i_idx])
        except Exception as e_bbox:
            print(f"ERROR obteniendo BBox para tag {surface_tags[i_idx]}: {e_bbox}")
            bbox_list[i_idx] = [0]*6 # Poner un valor por defecto para evitar errores posteriores


    # Ya tenemos las bounding boxes, ahora identificamos cada superficie
    epsilon = 1e-5 # Tolerancia para la planitud de las caras
    print("\n--- Identificación de Superficies ---")
    for i_idx in range(len(bbox_list)): 
        current_surface_tag = surface_tags[i_idx]
        current_bb = bbox_list[i_idx] 
        
        identity_found_for_surface_i = False 

        # --- Comprobación de Paredes ---
        if abs(current_bb[2] - current_bb[5]) < epsilon: 
            if abs(current_bb[2] - 0) < epsilon: 
                tags_superficies_nombradas["inferior"] = current_surface_tag
                identity_found_for_surface_i = True
            elif abs(current_bb[2] - Lz) < epsilon: 
                tags_superficies_nombradas["superior"] = current_surface_tag
                identity_found_for_surface_i = True
        
        if not identity_found_for_surface_i and abs(current_bb[0] - current_bb[3]) < epsilon: 
            if abs(current_bb[0] - 0) < epsilon: 
                tags_superficies_nombradas["trasera"] = current_surface_tag
                identity_found_for_surface_i = True
            elif abs(current_bb[0] - Lx) < epsilon: 
                tags_superficies_nombradas["frontal"] = current_surface_tag
                identity_found_for_surface_i = True

        if not identity_found_for_surface_i and abs(current_bb[1] - current_bb[4]) < epsilon: 
            if abs(current_bb[1] - 0) < epsilon: 
                tags_superficies_nombradas["izquierda"] = current_surface_tag
                identity_found_for_surface_i = True
            elif abs(current_bb[1] - Ly) < epsilon: 
                tags_superficies_nombradas["derecha"] = current_surface_tag
                identity_found_for_surface_i = True
        
        # --- Comprobación de Fuente ---
        if not identity_found_for_surface_i:
            is_flat_X = abs(current_bb[0] - current_bb[3]) < epsilon
            is_flat_Y = abs(current_bb[1] - current_bb[4]) < epsilon
            is_flat_Z = abs(current_bb[2] - current_bb[5]) < epsilon

            dim_x_bb = current_bb[3] - current_bb[0]
            dim_y_bb = current_bb[4] - current_bb[1]
            dim_z_bb = current_bb[5] - current_bb[2]
            expected_diameter = 2 * r_esfera
            diameter_epsilon = r_esfera * 0.5 

            is_sphere_dims = (abs(dim_x_bb - expected_diameter) < diameter_epsilon and
                            abs(dim_y_bb - expected_diameter) < diameter_epsilon and
                            abs(dim_z_bb - expected_diameter) < diameter_epsilon)

            if not is_flat_X and not is_flat_Y and not is_flat_Z and is_sphere_dims:
                if tags_superficies_nombradas["fuente"] == -1:
                    tags_superficies_nombradas["fuente"] = current_surface_tag
                else:
                    print(f"  ADVERTENCIA: Tag {current_surface_tag} parece ser otra fuente, pero 'fuente' ya estaba asignada a {tags_superficies_nombradas['fuente']}.")


    print("\n--- Tags de Superficies Identificadas (Resultado Final) ---")
    for nombre, tag_id in tags_superficies_nombradas.items():
        if tag_id != -1:
            print(f"Superficie '{nombre}': tag geométrico {tag_id}")
        else:
            print(f"Superficie '{nombre}': NO IDENTIFICADA (tag geométrico {tag_id})")

    # --- Asignar Grupos Físicos ---
    physical_tags_map = {
        "superior": 1,
        "inferior": 2,
        "frontal": 3,
        "trasera": 4,
        "derecha": 5,
        "izquierda": 6,
        "fuente": 7
    }
    volumen_phys_tag = 100

    print("\n--- Asignando Grupos Físicos ---")
    missing_surfaces = False
    for nombre_superficie, geom_tag_superficie in tags_superficies_nombradas.items():
        if geom_tag_superficie != -1:
            if nombre_superficie in physical_tags_map:
                phys_tag_para_esta_superficie = physical_tags_map[nombre_superficie]
                gmsh.model.addPhysicalGroup(2, [geom_tag_superficie], name=nombre_superficie, tag=phys_tag_para_esta_superficie)
                print(f"Grupo Físico de Superficie '{nombre_superficie}' (Tag Físico: {phys_tag_para_esta_superficie}) asignado a la entidad geométrica {geom_tag_superficie}.")
            else:
                print(f"ADVERTENCIA: La superficie '{nombre_superficie}' ({geom_tag_superficie}) fue identificada pero no tiene un tag físico predefinido en 'physical_tags_map'.")
        else:
            print(f"ERROR CRÍTICO: No se pudo identificar la superficie geométrica para '{nombre_superficie}', no se creará Grupo Físico.")
            missing_surfaces = True

    if missing_surfaces:
        print("ERROR FATAL: Faltan identificaciones de superficies geométricas. Abortando antes de mallar.")
        gmsh.finalize()
        sys.exit()


    gmsh.model.addPhysicalGroup(3, [volume_tag], name="dominio_volumetrico", tag=volumen_phys_tag)
    print(f"Grupo Físico de Volumen 'dominio_volumetrico' (Tag Físico: {volumen_phys_tag}) asignado a la entidad geométrica {volume_tag}.")


    # --- Configuración y Generación de Malla ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Netgen



    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_lc)
    print(f"Tamaño característico de malla: Min={min_lc}, Max={max_lc}")

    # Mantenemos orden 2 para la geometría de la esfera
    gmsh.model.mesh.setOrder(2)
    print("Establecido orden de la malla a 2 (elementos cuadráticos).")

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    print("Establecida versión del archivo MSH a 4.1")

    print("Generando malla 3D de segundo orden...")
    gmsh.model.mesh.generate(3)
    print("Malla generada.")

    # *** CAMBIO: Nombre de archivo para malla refinada ***
    mesh_output_filename = os.path.join(output_directory, "esfera_en_paralelepipedo_refined.msh")
    print(f"Guardando malla en: {mesh_output_filename}")
    gmsh.write(mesh_output_filename)
    print("Malla guardada.")

    # Descomenta si quieres ver la malla antes de salir
    #gmsh.fltk.run() 

    gmsh.finalize()
    print("Gmsh finalizado.")

    '''

