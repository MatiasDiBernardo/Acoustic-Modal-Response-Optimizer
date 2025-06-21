# Contenido para Source_image.py

import numpy as np
import pyroomacoustics as pra

def source_image_cuboid(room_dimensions, source_position, receptor_position, source_order, fs):
    """
    Genera la Respuesta al Impulso de la Sala (RIR) usando pyroomacoustics.
    Esta versión final convierte 'fs' a entero para evitar errores de tipo.
    """
    try:
        room = pra.ShoeBox(
            room_dimensions,
            fs=int(fs),  # <--- CORRECCIÓN CLAVE: Asegurarse de que fs sea un entero
            materials=None,
            max_order=source_order,
            use_rand_ism=False
        )

        # Formato robusto para la posición del micrófono
        mic_locs = np.array(receptor_position).reshape(3, 1)
        
        room.add_microphone(mic_locs)
        room.add_source(position=source_position)

        # Calcula la RIR
        room.compute_rir()
        
        # Extrae la RIR para el único micrófono y la única fuente
        rir = room.rir[0][0]

        return rir

    except Exception as e:
        # Imprime el error de forma más clara para facilitar el debugging
        print(f"Error dentro de source_image_cuboid: {e}")
        # Muestra el traceback completo para más detalles
        import traceback
        traceback.print_exc()
        return None