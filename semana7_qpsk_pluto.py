
import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from scipy import signal

# ==========================================
# 1. FUNCIONES BASE (Fase A)
# ==========================================
def generar_pn9(longitud=510):
    """
    Genera secuencia PN-9. Para QPSK necesitamos un número PAR de bits
    para poder agruparlos de 2 en 2 sin que sobre ninguno (510 bits = 255 símbolos).
    """
    registro = [1] * 9
    bits = []
    for _ in range(longitud):
        nuevo_bit = registro[4] ^ registro[8]
        bits.append(registro[-1])
        registro = [nuevo_bit] + registro[:-1]
    return np.array(bits)

# ==========================================
# 2. MODULADOR QPSK Y MAPEO DE CONSTELACIÓN
# ==========================================
print("1. Generando secuencia PN-9 (Par) y agrupar en símbolos QPSK...")
bits_tx = generar_pn9(510)
muestras_por_simbolo = 100

# Agrupamos los bits de 2 en 2: (bit_par, bit_impar)
simbolos_tx = []
for i in range(0, len(bits_tx), 2):
    b1 = bits_tx[i]
    b2 = bits_tx[i+1]
    # Mapeo a los 4 ángulos de fase (Cuadrantes I/Q con amplitud 1.0)
    if b1 == 0 and b2 == 0:
        simbolos_tx.append(np.exp(1j * np.pi / 4))       # 45 grados  (+I, +Q)
    elif b1 == 0 and b2 == 1:
        simbolos_tx.append(np.exp(1j * 3 * np.pi / 4))   # 135 grados (-I, +Q)
    elif b1 == 1 and b2 == 1:
        simbolos_tx.append(np.exp(1j * 5 * np.pi / 4))   # 225 grados (-I, -Q)
    elif b1 == 1 and b2 == 0:
        simbolos_tx.append(np.exp(1j * 7 * np.pi / 4))   # 315 grados (+I, -Q)

simbolos_tx = np.array(simbolos_tx)

# Aplicamos sobremuestreo para que el hardware transmita pulsos continuos
senal_qpsk_tx = np.repeat(simbolos_tx, muestras_por_simbolo)

# Escalamos al rango del DAC de 16 bits del ADALM-Pluto
escala_dac = 2**14
senal_tx_compleja = (senal_qpsk_tx * escala_dac).astype(np.complex64)

# ==========================================
# 3. CONFIGURACIÓN DEL ADALM-PLUTO (Fase B)
# ==========================================
print("2. Configurando ADALM-Pluto SDR (Modo Inalámbrico con Antenas)...")
try:
    sdr = adi.Pluto('ip:192.168.2.1')
except Exception as e:
    print(f"Error: No se encontró el Pluto. Detalle: {e}")
    exit()

sdr.sample_rate = int(1e6)          # 1 Msps
sdr.tx_lo = int(915e6)              # 915 MHz
sdr.rx_lo = int(915e6)              # 915 MHz

# Ganancias equilibradas
sdr.tx_hardwaregain_chan0 = -40     
sdr.rx_hardwaregain_chan0 = 20      

sdr.rx_buffer_size = len(senal_tx_compleja) * 4 
sdr.tx_cyclic_buffer = True

print("3. Transmitiendo QPSK y capturando del aire...")
sdr.tx(senal_tx_compleja)
time.sleep(1) # Estabilización de RF

for _ in range(5): _ = sdr.rx() # Limpiar búfer viejo
senal_rx_compleja = sdr.rx()

# Normalizamos la amplitud general para compensar pérdidas del aire
senal_rx_norm = senal_rx_compleja / np.mean(np.abs(senal_rx_compleja))

# ==========================================
# 4. SINCRONIZACIÓN Y DEMODULACIÓN QPSK (Fase C)
# ==========================================
print("4. Sincronizando secuencia en el tiempo...")
# Correlación usando la magnitud de los saltos de fase
correlacion = signal.correlate(np.abs(np.diff(senal_rx_norm)), np.abs(np.diff(senal_qpsk_tx)), mode='valid')
indice_inicio = np.argmax(correlacion)

rx_sinc = senal_rx_norm[indice_inicio : indice_inicio + len(senal_qpsk_tx)]

# Extraemos 1 sola muestra por símbolo justo en el centro (Muestreo óptimo)
simbolos_rx = []
for i in range(len(simbolos_tx)):
    centro = int((i + 0.5) * muestras_por_simbolo)
    simbolos_rx.append(rx_sinc[centro])

simbolos_rx = np.array(simbolos_rx)

print("5. Demodulando por decisión de cuadrantes I/Q...")
bits_rx = []
for s in simbolos_rx:
    # Evaluamos el signo de la parte Real (I) e Imaginaria (Q)
    I = np.real(s)
    Q = np.imag(s)
    
    if I >= 0 and Q >= 0:
        bits_rx.extend([0, 0]) # Cuadrante 1
    elif I < 0 and Q >= 0:
        bits_rx.extend([0, 1]) # Cuadrante 2
    elif I < 0 and Q < 0:
        bits_rx.extend([1, 1]) # Cuadrante 3
    elif I >= 0 and Q < 0:
        bits_rx.extend([1, 0]) # Cuadrante 4

bits_rx = np.array(bits_rx)

# ==========================================
# 5. CÁLCULO DE BER (Fase D)
# ==========================================
errores = np.sum(bits_tx != bits_rx)
ber = errores / len(bits_tx)

print("\n" + "="*50)
print(" RESULTADOS DE DEMODULACIÓN QPSK (Fase C & D)")
print("="*50)
print(f"  Total de bits evaluados (PN-9):  {len(bits_tx)} bits ({len(simbolos_tx)} símbolos)")
print(f"  Bits erróneos recibidos:         {errores}")
print(f"  TASA DE ERROR DE BIT (BER):      {ber:.6f}")
if ber < 1e-3:
    print("  ESTADO: ¡CUMPLE ESPECIFICACIÓN IEEE! (BER < 1e-3)")
else:
    print("  ESTADO: Desfase de portadora detectado. Revisa la constelación.")
print("="*50 + "\n")

# ==========================================
# 6. GRÁFICAS DE COMPROBACIÓN
# ==========================================
plt.figure(figsize=(14, 6))

# Subgráfica 1: El famoso Diagrama de Constelación I/Q
plt.subplot(1, 2, 1)
plt.scatter(np.real(simbolos_rx), np.imag(simbolos_rx), c='red', alpha=0.6, label='Símbolos RX (Aire)')
plt.scatter(np.real(simbolos_tx), np.imag(simbolos_tx), c='blue', marker='x', s=100, linewidths=3, label='Símbolos TX (Ideales)')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.title(f'Constelación QPSK (Plano I/Q) | BER = {ber:.4f}')
plt.xlabel('Componente en Fase (I)')
plt.ylabel('Componente en Cuadratura (Q)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.axis('equal') # Mantener proporción cuadrada exacta

# Subgráfica 2: Comparativa temporal de los primeros 30 bits (15 símbolos)
bits_ver = 30
muestras_ver = bits_ver * int(muestras_por_simbolo / 2)
tiempo = np.arange(muestras_ver) / sdr.sample_rate * 1000

plt.subplot(1, 2, 2)
plt.step(tiempo, np.repeat(bits_tx[:bits_ver], int(muestras_por_simbolo/2)), 'b-', linewidth=2, where='post', label='Bits TX')
plt.step(tiempo, np.repeat(bits_rx[:bits_ver], int(muestras_por_simbolo/2)), 'k--', linewidth=1.5, where='post', label='Bits RX')
plt.title('Comparativa Temporal de Flujo de Bits (2 Bits por Símbolo)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Nivel Lógico')
plt.ylim([-0.2, 1.3])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Liberación obligatoria de memoria
sdr.tx_destroy_buffer()
print("Memoria del ADALM-Pluto liberada correctamente.")