
import numpy as np
import matplotlib.pyplot as plt
import adi
import time

# ==========================================
# 1. FUNCIONES BASE DE LA FASE A
# ==========================================
def generar_pn9(longitud=511):
    """Genera la secuencia pseudoaleatoria PN-9 de 511 bits conocida."""
    registro = [1] * 9
    bits = []
    for _ in range(longitud):
        nuevo_bit = registro[4] ^ registro[8]
        bits.append(registro[-1])
        registro = [nuevo_bit] + registro[:-1]
    return np.array(bits)

def modulador_ask(bits, muestras_por_bit=100, amplitud=1.0):
    """Modula los bits en amplitud (On-Off) con sobremuestreo."""
    return np.repeat(bits, muestras_por_bit) * amplitud

# ==========================================
# 2. GENERACIÓN Y PREPARACIÓN DE LA SEÑAL
# ==========================================
print("1. Generando secuencia PN-9 y modulando en ASK...")
bits_tx = generar_pn9(511) # Secuencia completa de 511 bits
muestras_por_bit = 100     # Oversampling de 100 muestras
senal_ask = modulador_ask(bits_tx, muestras_por_bit=muestras_por_bit)

# ADVERTENCIA DE HARDWARE: El DAC del Pluto SDR trabaja con números complejos 
# enteros de 16 bits (-32768 a 32767). Debemos convertir nuestra señal flotante
# (0.0 a 1.0) en números complejos escalados para que la antena emita potencia.
escala_dac = 2**14 # Usamos 16384 para no saturar el transmisor
senal_tx_compleja = (senal_ask * escala_dac).astype(np.complex64)

# ==========================================
# 3. CONFIGURACIÓN DEL ADALM-PLUTO
# ==========================================
print("2. Conectando y configurando el ADALM-Pluto SDR...")
try:
    sdr = adi.Pluto('ip:192.168.2.1')
except Exception as e:
    print(f"Error de conexión con el Pluto: {e}")
    print("Asegúrate de que el cable USB esté conectado y la IP sea correcta.")
    exit()

# Asignación de especificaciones técnicas (con ajuste anti-saturación para antenas)
sdr.sample_rate = int(1e6)          # Sample rate: 1 Msps
sdr.tx_lo = int(915e6)              # Frecuencia portadora TX: 915 MHz (Banda ISM)
sdr.rx_lo = int(915e6)              # Frecuencia portadora RX: 915 MHz (Banda ISM)

# AJUSTE ANTI-SATURACIÓN (Reemplazamos los -10 dBm y 50 dB por potencias seguras):
sdr.tx_hardwaregain_chan0 = -40     # Bajamos TX a -40 dBm para no saturar el receptor al usar antenas
sdr.rx_hardwaregain_chan0 = 30      # Bajamos RX a 30 dB para capturar la onda limpia

# Configuración de buffers
sdr.rx_buffer_size = len(senal_tx_compleja) * 2 
sdr.tx_cyclic_buffer = True         # Transmisión cíclica para emisión continua

# ==========================================
# 4. TRANSMISIÓN Y RECEPCIÓN INALÁMBRICA
# ==========================================
print("3. Enviando señal al buffer de transmisión (TX)...")
sdr.tx(senal_tx_compleja) # El Pluto empieza a transmitir sin parar en 915 MHz

print("   Esperando estabilización del hardware...")
time.sleep(1) # Le damos 1 segundo al hardware para estabilizar el enlace RF

print("4. Capturando señal desde el puerto de recepción (RX)...")
# Limpiamos buffers viejos leyendo un par de veces al vacío
for _ in range(5):
    _ = sdr.rx()

# Captura real
senal_rx_compleja = sdr.rx()
# Para ASK, la información está en la magnitud (energía) de la señal recibida
senal_rx_magnitud = np.abs(senal_rx_compleja)

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================
print("5. Generando gráfica de validación experimental...")

# Graficamos solo los primeros 15 bits para poder apreciar claramente la forma de onda
muestras_a_ver = 15 * muestras_por_bit
tiempo = np.arange(muestras_a_ver) / sdr.sample_rate * 1000 # Tiempo en milisegundos

plt.figure(figsize=(12, 6))

# Subgráfica 1: Lo que enviamos a la antena
plt.subplot(2, 1, 1)
plt.plot(tiempo, senal_ask[:muestras_a_ver], 'b-', linewidth=2, label='TX: Señal ASK Ideal (Enviada)')
plt.title('Fase B: Validación de Transmisión y Recepción (ADALM-Pluto con Antenas)')
plt.ylabel('Amplitud TX')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.ylim([-0.2, 1.3])

# Subgráfica 2: Lo que capturó físicamente el receptor
plt.subplot(2, 1, 2)
plt.plot(tiempo, senal_rx_magnitud[:muestras_a_ver], 'r-', linewidth=1.5, label='RX: Magnitud Recibida (Física)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Magnitud RX (ADC)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("¡Prueba de Fase B completada con éxito!")