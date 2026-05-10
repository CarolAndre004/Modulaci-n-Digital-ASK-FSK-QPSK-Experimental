import numpy as np

def generar_pn9(longitud=511):

    registro = [1] * 9
    bits = []

    for _ in range(longitud):

        nuevo_bit = registro[4] ^ registro[8]

        bits.append(registro[-1])

        registro = [nuevo_bit] + registro[:-1]

    return np.array(bits)

if __name__ == "__main__":

    secuencia = generar_pn9()

    print(secuencia)