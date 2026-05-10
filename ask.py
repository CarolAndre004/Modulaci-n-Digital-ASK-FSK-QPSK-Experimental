import numpy as np
import matplotlib.pyplot as plt

# =====================
# GENERADOR PN9
# =====================
def generar_pn9(longitud=511):

    registro = [1] * 9
    bits = []

    for _ in range(longitud):

        nuevo_bit = registro[4] ^ registro[8]

        bits.append(registro[-1])

        registro = [nuevo_bit] + registro[:-1]

    return np.array(bits)


# =====================
# MODULADOR ASK
# =====================
def modulador_ask(bits, muestras_por_bit=100, amplitud=1):

    señal = []

    for bit in bits:

        if bit == 1:
            simbolo = amplitud * np.ones(muestras_por_bit)

        else:
            simbolo = np.zeros(muestras_por_bit)

        señal.extend(simbolo)

    return np.array(señal)


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    bits = generar_pn9(20)

    señal_ask = modulador_ask(bits)

    print("Bits transmitidos:")
    print(bits)

    plt.plot(señal_ask)
    plt.title("Modulación ASK")
    plt.grid()
    plt.show()