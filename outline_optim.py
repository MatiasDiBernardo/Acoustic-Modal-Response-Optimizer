import numpy as np

from room.geometry_generator import calculation_of_geometry_simple
from FEM.mode_sumation import compute_modal_sum_average
from aux.merit_figure import merit_spatial_deviation, merit_magnitude_deviation

def calculate_new_dimensiones(L_best, L_original, wcs):
    if L_best + wcs < L_original:
        return L_best
    else:
        return L_original

def calculate_new_spacing(L_best, L_original, spacing_original):
    center_diff = (L_original - L_best)/2
    dx = (L_best + center_diff) - (L_original - spacing_original)
    return dx

def recalculate_spatial_dimensions(best_room, Lx, Ly, Lz, Dx, Dy, Dz):
    wcs = 0.05  # Worst case spacing
    Lx_best, Ly_best, Lz_best = best_room
    
    Lx_final = calculate_new_dimensiones(Lx_best, Lx, wcs)
    Dx_final = calculate_new_spacing(Lx_final, Lx, Dx)

    Ly_final = calculate_new_dimensiones(Ly_best, Ly, wcs)
    Dy_final = calculate_new_spacing(Ly_final, Ly, Dy)

    final_best_room = (Lx_final, Ly_final, Lz_best)
    final_spacing = (Dx_final, Dy_final)
    
    return final_best_room, final_spacing

def find_best_outline(Lx, Ly, Lz, Dx, Dy, Dz, source_position, receptor_position, optim_type):
    """Encuentra el mejor cuarto paralelepipedo en base a las dimensiones del cuarto y el
    margen seleccionado. Utiliza SM para el calculo de respuesta en frecuencia.

    Args:
        Lx (float): Ancho en metros
        Ly (float): Largo en metros
        Lz (float): Alto en metros
        Dx (float): Espaciado del ancho en metros
        Dy (float): Espaciado del largo en metros
        Dz (float): Espaciado del alto en metros
        source_position (tuple(x, y, z)): Posicion de la fuente en metros
        receptor_position (tuple(x, y, z)): Posicion del receptor en metros
        optim_type: Tipo de optimización, puede ser "slow", "medium" or "fast"
    Returns:
        best_room (tuple(x, y, z)): Dimensiones del mejor cuarto encontrado 
        spacing_room (tuple(dx, dy, dz)): Espaciado del mejor cuarto encontrado 
        merit_figure (tuple(FM, MD, SD)): Figuras de mérito
        mag (array): Respuesta en frecuencia del mejor cuarto
    """

    # Controles del proceso
    freqs = np.arange(20, 200, 1)  # Rango de frecuencias para el modal sum
    
    # Salas a iterar
    if optim_type == "Fast":
        initial_rooms = 500
    if optim_type == "Medium":
        initial_rooms = 1000
    if optim_type == "Slow":
        initial_rooms = 2000
    
    # Resultados
    merit_sv_values = []
    merit_md_values = []
    mag_responses = []
    rooms = []

    ## Optimización
    for _ in range(initial_rooms):
        Lx_new, Ly_new, Lz_new = calculation_of_geometry_simple(Lx, Ly, Lz, Dx, Dy, Dz)
        dx_room = (Lx - Lx_new)/2
        dy_room = (Ly - Ly_new)/2
        new_source_pos = (source_position[0] - dx_room, source_position[1] - dy_room, source_position[2])
        new_receptor_pos = (receptor_position[0] - dx_room, receptor_position[1] - dy_room, receptor_position[2])

        mag = compute_modal_sum_average((Lx_new, Ly_new, Lz_new), new_source_pos, new_receptor_pos, freqs)
        sv_merit = merit_spatial_deviation(mag)
        md_merit = merit_magnitude_deviation(mag)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_avg = np.sum(mag, axis=0)/7
        mag_responses.append(mag_avg)
        rooms.append((Lx_new, Ly_new, Lz_new))
    
    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    idx_best_room = np.argmin(merit_general)
    final_best_room, best_room_spacing = recalculate_spatial_dimensions(rooms[idx_best_room],Lx, Ly, Lz, Dx, Dy, Dz) 

    mag_best = compute_modal_sum_average(final_best_room, source_position, receptor_position, freqs)
    sv_merit = merit_spatial_deviation(mag_best)
    md_merit = merit_magnitude_deviation(mag_best)
    merit = (sv_merit + md_merit, md_merit, sv_merit)
    mag = np.sum(mag_best, axis=0)/7

    return final_best_room, best_room_spacing, merit, mag
