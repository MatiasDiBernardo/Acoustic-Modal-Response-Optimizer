import numpy as np
import matplotlib.pyplot as plt
from FEM_source import FEM_Source_Solver_Spatial_Average, FEM_Source_Solver

def compute_modal_transfer(rs, rr, L, freqs, c=343.0, eta=0.01):
    """
    Computes the frequency-domain transfer function H(f) between source rs and receiver rr
    in a rigid rectangular room via analytic modal summation. List of assumptions for this
    model to be valid. This formulation simplyfies damping and ignores free field solution.
    - Monopole point source of unit strength
    - Rigid, lossless walls (Neumann boundary conditions)
    - No scattering objects or furniture inside the room
    - Homogeneous, isotropic medium

    Parameters:
    - rs: array-like of shape (3,), source coordinates [x, y, z]
    - rr: array-like of shape (3,), receiver coordinates [x, y, z]
    - L: tuple (Lx, Ly, Lz), room dimensions
    - freqs: 1D array of frequencies (Hz) at which to compute H
    - eta: float or array, modal damping factor (default 0.01)
    - c: float, speed of sound in m/s (default 343.0)

    Returns:
    - H: 1D complex array, transfer function at each frequency
    """
    rs = np.asarray(rs)
    rr = np.asarray(rr)
    Lx, Ly, Lz = L
    omega = 2 * np.pi * freqs
    f_max = max(freqs)
    
    # Determine max mode indices for each axis
    n_max_x = int(2 * Lx * f_max / c) + 1
    n_max_y = int(2 * Ly * f_max / c) + 1
    n_max_z = int(2 * Lz * f_max / c) + 1
    
    H = np.zeros_like(freqs, dtype=complex)
    
    for nx in range(n_max_x + 1):
        for ny in range(n_max_y + 1):
            for nz in range(n_max_z + 1):
                if nx == ny == nz == 0:
                    continue  # skip trivial mode
                
                # eigenfrequency
                omega_n = np.pi * c * np.sqrt((nx / Lx)**2 + (ny / Ly)**2 + (nz / Lz)**2)
                f_n = omega_n / (2 * np.pi)
                
                if f_n > f_max:
                    continue  # skip modes above f_max
                
                # eigenfunctions at source and receiver
                phi_s = (np.cos(nx * np.pi * rs[0] / Lx) *
                         np.cos(ny * np.pi * rs[1] / Ly) *
                         np.cos(nz * np.pi * rs[2] / Lz))
                phi_r = (np.cos(nx * np.pi * rr[0] / Lx) *
                         np.cos(ny * np.pi * rr[1] / Ly) *
                         np.cos(nz * np.pi * rr[2] / Lz))
                
                # modal contribution
                H += (phi_s * phi_r) / (omega_n**2 - omega**2 + 2j * eta * omega_n * omega)
    
    rta = 20 * np.log10(np.abs(H))
    return rta

import numpy as np

def compute_modal_transfer_complete(
    rs, rr, L, freqs,
    c: float = 343.0,                # speed of sound [m/s]
    rho0: float = 1.21,              # air density [kg/m³] at 20 °C
    S0: float = 1.0,                 # monopole “area” (unit strength)
    eps: tuple = (0.2, 0.2, 0.2),    # wall absorption coeffs (εx, εy, εz)
    betas: tuple = (1.0, 1.0, 1.0),   # mode‑weight factors (βx, βy, βz)
    include_direct: bool = True
) -> np.ndarray:
    """
    Compute H(f) via the paper’s modal‐decomposition (Eqs. 6–11),
    with default physical parameters for a typical room.

    Returns the transfer function on dB.
    """

    rs = np.asarray(rs)
    rr = np.asarray(rr)
    Lx, Ly, Lz = L
    omega = 2 * np.pi * freqs
    f_max = freqs.max()

    # Preallocate
    H_modal = np.zeros_like(freqs, dtype=complex)

    # max mode indices per axis
    n_max = lambda Ldim: int(2 * Ldim * f_max / c) + 1
    Nxs, Nys, Nzs = map(n_max, (Lx, Ly, Lz))

    eps_x, eps_y, eps_z = eps
    βx, βy, βz     = betas

    for nx in range(Nxs + 1):
        for ny in range(Nys + 1):
            for nz in range(Nzs + 1):
                if nx == ny == nz == 0:
                    continue

                # eigenfrequency ω_n and cutoff
                omega_n = np.pi * c * np.sqrt((nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2)
                f_n = omega_n / (2 * np.pi)
                if f_n > f_max:
                    continue

                # eigenfunctions at rs, rr
                phi_s = (np.cos(nx*np.pi*rs[0]/Lx) *
                         np.cos(ny*np.pi*rs[1]/Ly) *
                         np.cos(nz*np.pi*rs[2]/Lz))
                phi_r = (np.cos(nx*np.pi*rr[0]/Lx) *
                         np.cos(ny*np.pi*rr[1]/Ly) *
                         np.cos(nz*np.pi*rr[2]/Lz))

                # numerator A_n = j·ρ0·S0·c²·φ_s·φ_r
                A_n = 1j * rho0 * S0 * c**2 * (phi_s * phi_r)

                # modal damping δ_n per Eq. (11)
                delta_n = (c/omega_n) * (
                    eps_x * βx / Lx +
                    eps_y * βy / Ly +
                    eps_z * βz / Lz
                )

                # denominator exactly as paper (ω² − ω_n² − j2ω_nδ_nω)
                denom = (omega**2 - omega_n**2) - 1j * 2 * omega_n * delta_n * omega

                H_modal += A_n / denom

    if include_direct:
        # add free‐field monopole path
        R = np.linalg.norm(rr - rs)
        direct = np.exp(-1j * omega * R) / (4 * np.pi * R)
        rta = H_modal + direct
        rta_db = 20 * np.log10(np.abs(rta))
        return rta_db

    rta_db = 20 * np.log10(np.abs(H_modal))
    return rta_db

# Parametros
L = (3.0, 4.0, 2.2)          # Room dimensions [m]
rs = [1.0, 1.5, 1.2]        # Source position [m]
rr = [3.0, 2.0, 1.8]        # Receiver position [m]
freqs = np.arange(20, 200, 1)

# Modal Sum
rta = compute_modal_transfer(rs, rr, L, freqs)

# FEM
mesh = "mallado/room_max200.msh"  # Crear malla con el script correspondiente
res_fem = FEM_Source_Solver(freqs, mesh, rr)
#space_average = 6  # El agregar promediado espacial casi no afecta la preformance
# res_base = FEM_Source_Solver_Spatial_Average(freqs, mesh, rr, space_average)

# Plot the magnitude of the transfer function
# amp_dif = res_base[0][0] - rta[0]
# rta += amp_dif

# Normalizo el inicio de las dos respuestas
rta -= rta[0]
res_fem = 20 * np.log10(np.abs(res_fem))
res_fem -= res_fem[0]

plt.figure()
plt.plot(freqs, rta, label="Modal Summation")
#plt.plot(freqs, res_base[0], label="FEM")
plt.plot(freqs, res_fem, label="FEM")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.show()
