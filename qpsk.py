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
# MODULADOR QPSK
# =====================
def modulador_qpsk(bits):

    # Si hay número impar, agrega un 0
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    simbolos = []

    for i in range(0, len(bits), 2):

        b1 = bits[i]
        b2 = bits[i+1]

        if b1 == 0 and b2 == 0:
            simbolo = 1 + 1j

        elif b1 == 0 and b2 == 1:
            simbolo = -1 + 1j

        elif b1 == 1 and b2 == 0:
            simbolo = 1 - 1j

        else:
            simbolo = -1 - 1j

        simbolos.append(simbolo)

    return np.array(simbolos)


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    bits = generar_pn9()

    simbolos = modulador_qpsk(bits)

    print("Símbolos QPSK:")
    print(simbolos)

    plt.scatter(np.real(simbolos), np.imag(simbolos))
    plt.title("Constelación QPSK")
    plt.grid()
    plt.show()