import numpy as np
import matplotlib.pyplot as plt


# =====================
# GENERADOR PN9
# =====================
def generar_pn9(longitud=100):

    registro = [1] * 9
    bits = []

    for _ in range(longitud):

        nuevo_bit = registro[4] ^ registro[8]

        bits.append(registro[-1])

        registro = [nuevo_bit] + registro[:-1]

    return np.array(bits)


# =====================
# CANAL CON RUIDO
# =====================
def agregar_ruido(bits, prob_error=0.1):

    bits_ruidosos = bits.copy()

    for i in range(len(bits)):

        if np.random.rand() < prob_error:

            bits_ruidosos[i] = 1 - bits_ruidosos[i]

    return bits_ruidosos


# =====================
# BER
# =====================
def calcular_ber(original, recibido):

    errores = np.sum(original != recibido)

    ber = errores / len(original)

    return ber


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    bits_tx = generar_pn9()

    bits_rx = agregar_ruido(bits_tx)

    ber = calcular_ber(bits_tx, bits_rx)

    print("BER =", ber)