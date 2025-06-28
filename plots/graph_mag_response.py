import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

def general_mag_response(freqs, mag):
    plt.figure("Resultado magnitud", figsize=(10, 4))
    plt.plot(freqs, mag)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud (dB)')
    plt.grid()
    plt.show()

def mag_response_comparison(list_mag):
    """List of mags to graph with (freq, mag, name)

    Args:
        list_mag (list): List where each element is (freq, mag, name)
    """
    plt.figure("Resultado magnitud", figsize=(10, 4))
    for val in list_mag:
        freq, mag, name = val
        plt.plot(freq, mag, '--', linewidth=2, label=name)

    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud (dB)')
    plt.legend()
    plt.grid()
    plt.show()