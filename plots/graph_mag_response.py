import numpy as np
import matplotlib.pyplot as plt

def general_mag_response(freqs, mag):
    plt.figure("Resultado magnitud", figsize=(10, 4))
    plt.plot(freqs, mag)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud (dB)')
    plt.grid()
    plt.show()