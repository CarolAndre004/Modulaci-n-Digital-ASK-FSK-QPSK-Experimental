import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from scipy import signal

# ==========================================
# 1. FUNCIONES BASE (Fase A)
# ==========================================
def generar_pn9(longitud=511):
    """Genera la secuencia pseudoaleatoria PN-9 conocida."""
    registro = [1] * 9
    bits = []
    for _ in range(longitud):
        nuevo_bit = registro[4] ^ registro[8]
        bits.append(registro[-1])
        registro = [nuevo_bit] + registro[:-1]
    return np.array(bits)

def modulador_ask(bits, muestras_por_bit=100, amplitud=1.0):
    """Modula en amplitud On-Off."""
    return np.repeat(bits, muestras_por_bit) * amplitud

# ==========================================
# 2. PREPARACIÓN Y TRANSMISIÓN (Fase B)
# ==========================================
print("1. Generando secuencia PN-9 y modulando en ASK...")
bits_tx = generar_pn9(511)
muestras_por_bit = 100
senal_ask_tx = modulador_ask(bits_tx, muestras_por_bit=muestras_por_bit)

escala_dac = 2**14
senal_tx_compleja = (senal_ask_tx * escala_dac).astype(np.complex64)

print("2. Configurando ADALM-Pluto SDR (Modo Inalámbrico con Antenas)...")
try:
    sdr = adi.Pluto('ip:192.168.2.1')
except Exception as e:
    print(f"Error: No se encontró el Pluto. Detalle: {e}")
    exit()

sdr.sample_rate = int(1e6)          # 1 Msps
sdr.tx_lo = int(915e6)              # 915 MHz
sdr.rx_lo = int(915e6)              # 915 MHz
sdr.tx_hardwaregain_chan0 = -40     # Potencia baja para evitar saturación en antenas
sdr.rx_hardwaregain_chan0 = 30      # Sensibilidad equilibrada

# Capturamos 3 veces el tamaño de la secuencia para garantizar que al menos una caiga completa
sdr.rx_buffer_size = len(senal_tx_compleja) * 3
sdr.tx_cyclic_buffer = True

print("3. Transmitiendo señal y capturando del aire...")
sdr.tx(senal_tx_compleja)
time.sleep(1) # Estabilización

for _ in range(5): _ = sdr.rx() # Limpiar búfer
senal_rx_compleja = sdr.rx()

# 1. DETECTOR DE ENERGÍA (Magnitud)
rx_magnitud = np.abs(senal_rx_compleja)

# ==========================================
# 3. DEMODULACIÓN Y SINCRONIZACIÓN (Fase C)
# ==========================================
print("4. Sincronizando secuencia en el tiempo mediante correlación...")

# Normalizamos para correlacionar sin que importe la ganancia o el piso de ruido
tx_norm = senal_ask_tx - np.mean(senal_ask_tx)
rx_norm = rx_magnitud - np.mean(rx_magnitud)

# Correlación cruzada para encontrar el inicio exacto del PN-9
correlacion = signal.correlate(rx_norm, tx_norm, mode='valid')
indice_inicio = np.argmax(correlacion)

# Recortamos exactamente la ventana de 511 bits sincronizada
rx_sincronizada = rx_magnitud[indice_inicio : indice_inicio + len(senal_ask_tx)]

print("5. Demodulando bits con umbral dinámico anti-ruido...")
bits_rx = []
energias_simbolos = []

for i in range(len(bits_tx)):
    # Extraemos el bloque de 100 muestras correspondiente a 1 bit
    bloque_bit = rx_sincronizada[i * muestras_por_bit : (i + 1) * muestras_por_bit]
    # Tomamos la mediana para ignorar los picos de ringing de los extremos
    energia_media = np.median(bloque_bit)
    energias_simbolos.append(energia_media)

# UMBRAL DINÁMICO: Mitad de camino entre el percentil 10 (ceros) y el percentil 90 (unos)
nivel_bajo = np.percentile(energias_simbolos, 10)
nivel_alto = np.percentile(energias_simbolos, 90)
umbral_decision = (nivel_alto + nivel_bajo) / 2

# Decisión lógica bit a bit
for energia in energias_simbolos:
    if energia > umbral_decision:
        bits_rx.append(1)
    else:
        bits_rx.append(0)

bits_rx = np.array(bits_rx)

# ==========================================
# 4. CÁLCULO DE BER (Fase D)
# ==========================================
errores = np.sum(bits_tx != bits_rx)
ber = errores / len(bits_tx)

print("\n" + "="*50)
print(" RESULTADOS DE DEMODULACIÓN ASK (Fase C & D)")
print("="*50)
print(f"  Total de bits evaluados (PN-9):  {len(bits_tx)} bits")
print(f"  Nivel medio detectado Bit 0:     {nivel_bajo:.2f} ADC")
print(f"  Nivel medio detectado Bit 1:     {nivel_alto:.2f} ADC")
print(f"  Umbral dinámico aplicado:        {umbral_decision:.2f} ADC")
print("-" * 50)
print(f"  Bits erróneos recibidos:         {errores}")
print(f"  TASA DE ERROR DE BIT (BER):      {ber:.6f}")
if ber < 1e-3:
    print("  ESTADO: ¡CUMPLE ESPECIFICACIÓN IEEE! (BER < 1e-3)")
else:
    print("  ESTADO: BER elevado. Revisa orientación de antenas o ganancia.")
print("="*50 + "\n")

# ==========================================
# 5. GRÁFICAS DE COMPROBACIÓN
# ==========================================
bits_ver = 20 # Ver los primeros 20 bits en la gráfica
muestras_ver = bits_ver * muestras_por_bit
tiempo = np.arange(muestras_ver) / sdr.sample_rate * 1000

plt.figure(figsize=(12, 8))

# Gráfica 1: Señal capturada y el Umbral Dinámico
plt.subplot(2, 1, 1)
plt.plot(tiempo, rx_sincronizada[:muestras_ver], 'r-', alpha=0.8, label='Señal RX Sincronizada')
plt.axhline(umbral_decision, color='green', linestyle='--', linewidth=2, label=f'Umbral de Decisión ({umbral_decision:.1f})')
plt.title(f'Demodulación ASK - Detector de Energía | BER Experimental = {ber:.4f}')
plt.ylabel('Magnitud RX (ADC)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

# Gráfica 2: Comparación Bit a Bit (TX vs RX)
plt.subplot(2, 1, 2)
plt.step(tiempo, np.repeat(bits_tx[:bits_ver], muestras_por_bit), 'b-', linewidth=2, where='post', label='Bits TX (Enviados)')
plt.step(tiempo, np.repeat(bits_rx[:bits_ver], muestras_por_bit), 'k--', linewidth=1.5, where='post', label='Bits RX (Recuperados)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Nivel Lógico')
plt.ylim([-0.2, 1.3])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("¡Demodulación completada!")