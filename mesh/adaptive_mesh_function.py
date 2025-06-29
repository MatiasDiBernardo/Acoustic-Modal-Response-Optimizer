
from mesh.mesh_3D_generator import generate_mesh_parallelepiped
import gmsh
import numpy as np
import os
# Asumimos que la siguiente función ya está importada y es funcional
# from mesh.mesh_3D_generator import generate_mesh_parallelepiped



def adaptive_mesh_generation(floor_coords, Z, source_position, f_max, f_min_global=20.0, mesh_number=3, elements_per_wavelength=10, name="room"):
    """
    Genera múltiples mallas y devuelve un diccionario con sus
    RUTAS ABSOLUTAS CORRECTAS y los rangos de frecuencia para los que son válidas.
    """
    print(f"--- Iniciando generación de {mesh_number} mallas adaptativas ---")
    mesh_info_dict = {}

    # --- INICIO DE MODIFICACIÓN: Construcción de Ruta Robusta ---
    # 1. Obtener la ruta del directorio donde se encuentra este script.
    #    '__file__' es una variable especial de Python que contiene esta información.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. Subir un nivel en la jerarquía de carpetas para llegar a la raíz del proyecto.
        #    Esto asume que este script está en una subcarpeta como /mesh o /FEM.
        project_root = os.path.dirname(script_dir)
    except NameError:
        # Fallback si se ejecuta interactivamente (ej. en un notebook)
        project_root = os.path.abspath('')
        print("Advertencia: Ejecutando en modo interactivo. Se asume que la raíz del proyecto es el directorio actual.")

    # 3. Construir la ruta absoluta y correcta a la carpeta 'mallado'.
    output_dir_abs = os.path.join(project_root, 'mallado')
    
    # 4. Asegurarse de que el directorio de salida exista.
    os.makedirs(output_dir_abs, exist_ok=True)
    print(f"Directorio de salida para las mallas: {output_dir_abs}")
    # --- FIN DE MODIFICACIÓN ---

    frequency_bands = np.linspace((f_max - f_min_global) / mesh_number, f_max, mesh_number)
    print(f"Frecuencias máximas para cada banda: {frequency_bands}")

    f_start = f_min_global
    for i, f_band_max in enumerate(frequency_bands):
        mesh_name = f"{name}_band_{i+1}_fmax_{int(f_band_max)}"
        print(f"\nGenerando malla {i+1}/{mesh_number}: '{mesh_name}.msh'")
        
        # Construir la ruta de salida completa para este archivo específico
        ruta_absoluta_malla = os.path.join(output_dir_abs, f"{mesh_name}.msh")

        # Es crucial que tu función de generación también sepa dónde guardar.
        # La forma más limpia es pasarle la ruta completa.
        generate_mesh_parallelepiped(
            floor_coords, Z, source_position, f_band_max,
            elements_per_wavelength, 
            name=mesh_name, # Puedes mantener 'name' si lo usas para otra cosa
            output_path=ruta_absoluta_malla # Se pasa la ruta completa
        )
        
        # Se guarda la misma ruta absoluta en el diccionario
        mesh_info_dict[ruta_absoluta_malla] = (f_start, f_band_max)
        f_start = f_band_max

    print(f"\n--- Generación de {mesh_number} mallas finalizada. ---")
    return mesh_info_dict
