import numpy as np

# Figura de merito sobre la distribución modal
def modal_response_merit(modal_response):
    """Calcula un valor en función al espaciado de los modos según la geometría.

    Args:
        modal_response (np.array): Array con la frecuencias donde caen los modos

    Returns:
        _type_: _description_
    """
    # Implementación naive de la figura de error
    distance_between_modes = modal_response[0:-1] - modal_response[1:]  # Diferencia entre modos consecutivos
    return np.std(distance_between_modes)

# También tiene que haber una figura de merito sobre la simulación con esfera pulsante (y diferentes posiciones de a sala)