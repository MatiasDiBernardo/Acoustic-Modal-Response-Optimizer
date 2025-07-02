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

def merit_spatial_deviation(response_matrix):
    std_per_column = np.std(response_matrix, axis=0, ddof=1)
    return np.mean(std_per_column)

def merit_magnitude_deviation(response_matrix):
    std_per_row = np.std(response_matrix, axis=1, ddof=1)
    return np.mean(std_per_row)

