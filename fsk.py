import numpy as np
import matplotlib.pyplot as plt


# =====================
# GENERADOR PN9
# =====================
def generar_pn9(longitud=20):

    registro = [1] * 9
    bits = []

    for _ in range(longitud):

        nuevo_bit = registro[4] ^ registro[8]

        bits.append(registro[-1])

        registro = [nuevo_bit] + registro[:-1]

    return np.array(bits)


# =====================
# MODULADOR FSK
# =====================
def modulador_fsk(bits, muestras_por_bit=100):

    f0 = 5
    f1 = 15

    señal = []

    t = np.linspace(0, 1, muestras_por_bit)

    for bit in bits:

        if bit == 0:

            simbolo = np.sin(2 * np.pi * f0 * t)

        else:

            simbolo = np.sin(2 * np.pi * f1 * t)

        señal.extend(simbolo)

    return np.array(señal)


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    bits = generar_pn9()

    señal_fsk = modulador_fsk(bits)

    print("Bits transmitidos:")
    print(bits)

    plt.plot(señal_fsk)
    plt.title("Modulación FSK")
    plt.grid()
    plt.show()