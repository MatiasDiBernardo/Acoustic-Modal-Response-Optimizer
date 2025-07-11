import sys
import os # Importado para crear el directorio de salida
import gmsh
import numpy as np

def create_simple_mesh(Lx, Ly, Lz, source_pos, f_max, name):
    """Crea una malla para una geometría simple.

    Args:
        Lx (int): Ancho en metros
        Ly (int): Largo en metros
        Lz (int): Altura en metros
        source_pos (tuple(x, y, z)): Posicion de la fuente
        f_max (float): Frecuencia máxima
        name (string): Nombre del archivo
    """
    gmsh.initialize()
    gmsh.model.add("room_lest_refined") # Nombre actualizado
    gmsh.option.setNumber("General.Terminal", 0)

    # Crear directorio de salida si no existe
    output_directory = "mallado"
    os.makedirs(output_directory, exist_ok=True)

    #Se define la frecuencia maxima de analisis, la cual va determinar la resolucion de la malla 
    c = 343
    landa_max = c/f_max

    # Se definen las dimensiones del paralelepípedo y la ubicación de la esfera (todo en S.I)
    # Paralelepípedo (dominio pequeño)

    # Fuente esférica
    x_esfera = source_pos[0]
    y_esfera = source_pos[1]
    z_esfera = source_pos[2]

    r_esfera = 0.1 # Radio de la esfera interior

    #Se definen las dimensiones maximas y minimas de los elementos
    # *** CAMBIO: Tamaños de malla refinados ***
    min_lc = r_esfera / 10   
    max_lc = landa_max / 10  

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

    # Obtener los límites (superficies) del volumen resultante
    contornos = gmsh.model.getBoundary(dominio, combined=False, oriented=False, recursive=False)

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

    missing_surfaces = False
    for nombre_superficie, geom_tag_superficie in tags_superficies_nombradas.items():
        if geom_tag_superficie != -1:
            if nombre_superficie in physical_tags_map:
                phys_tag_para_esta_superficie = physical_tags_map[nombre_superficie]
                gmsh.model.addPhysicalGroup(2, [geom_tag_superficie], name=nombre_superficie, tag=phys_tag_para_esta_superficie)
        else:
            print(f"ERROR CRÍTICO: No se pudo identificar la superficie geométrica para '{nombre_superficie}', no se creará Grupo Físico.")
            missing_surfaces = True

    if missing_surfaces:
        print("ERROR FATAL: Faltan identificaciones de superficies geométricas. Abortando antes de mallar.")
        gmsh.finalize()
        sys.exit()


    gmsh.model.addPhysicalGroup(3, [volume_tag], name="dominio_volumetrico", tag=volumen_phys_tag)

    # --- Configuración y Generación de Malla ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Netgen

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_lc)

    # Mantenemos orden 2 para la geometría de la esfera
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    gmsh.model.mesh.generate(3)

    mesh_output_filename = os.path.join(output_directory, f"{name}.msh")
    gmsh.write(mesh_output_filename)

    gmsh.finalize()