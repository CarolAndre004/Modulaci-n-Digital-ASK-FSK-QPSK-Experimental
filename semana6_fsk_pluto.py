
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

# ==========================================
# 2. MODULADOR FSK DE FASE CONTINUA (CPFSK)
# ==========================================
print("1. Generando secuencia PN-9 y modulando en FSK...")
bits_tx = generar_pn9(511)
muestras_por_bit = 100
sample_rate = int(1e6) # 1 Msps

# Repetimos los bits para el sobremuestreo
bits_repetidos = np.repeat(bits_tx, muestras_por_bit)

# Asignamos desviaciones de frecuencia:
# Bit 0 -> -100 kHz
# Bit 1 -> +100 kHz
desviacion_freq = 100e3 
frecuencia_instantanea = np.where(bits_repetidos == 1, desviacion_freq, -desviacion_freq)

# Integramos la frecuencia para obtener una fase continua sin saltos bruscos
fase = 2 * np.pi * np.cumsum(frecuencia_instantanea) / sample_rate

# Generamos la señal compleja en banda base (Amplitud constante = 1.0)
senal_fsk_tx = np.exp(1j * fase)

# Escalamos para el DAC del ADALM-Pluto (16 bits)
escala_dac = 2**14
senal_tx_compleja = (senal_fsk_tx * escala_dac).astype(np.complex64)

# ==========================================
# 3. CONFIGURACIÓN DEL ADALM-PLUTO (Fase B)
# ==========================================
print("2. Configurando ADALM-Pluto SDR (Modo Inalámbrico con Antenas)...")
try:
    sdr = adi.Pluto('ip:192.168.2.1')
except Exception as e:
    print(f"Error: No se encontró el Pluto. Detalle: {e}")
    exit()

sdr.sample_rate = sample_rate       # 1 Msps
sdr.tx_lo = int(915e6)              # 915 MHz
sdr.rx_lo = int(915e6)              # 915 MHz

# Mantener potencia segura para no saturar al usar antenas
sdr.tx_hardwaregain_chan0 = -40     
sdr.rx_hardwaregain_chan0 = 20      

sdr.rx_buffer_size = len(senal_tx_compleja) * 4 
sdr.tx_cyclic_buffer = True

print("3. Transmitiendo FSK y capturando del aire...")
sdr.tx(senal_tx_compleja)
time.sleep(1) # Estabilización de RF

for _ in range(5): _ = sdr.rx() # Limpiar búfer
senal_rx_compleja = sdr.rx()

# ==========================================
# 4. DEMODULADOR FSK POR DISCRIMINADOR DE FASE (Fase C)
# ==========================================
print("4. Demodulando por frecuencia instantánea...")

# 1. Extraemos el ángulo de fase de toda la señal capturada
fase_rx = np.unwrap(np.angle(senal_rx_compleja))

# 2. Derivamos la fase respecto al tiempo para obtener la FRECUENCIA (Hz)
# f = (1 / 2*pi) * (d_theta / dt)
freq_rx = np.diff(fase_rx) * sample_rate / (2 * np.pi)

# Suavizado ligero para eliminar ruido térmico del aire
ventana_suav = 15
freq_suavizada = np.convolve(freq_rx, np.ones(ventana_suav)/ventana_suav, mode='same')

print("5. Sincronizando secuencia en el tiempo...")
# Sincronizamos usando la frecuencia ideal transmitida vs la recibida
correlacion = signal.correlate(freq_suavizada, frecuencia_instantanea[:-1], mode='valid')
indice_inicio = np.argmax(correlacion)

freq_sincronizada = freq_suavizada[indice_inicio : indice_inicio + len(frecuencia_instantanea)]

# DECISIÓN LÓGICA: Como usamos -100 kHz para 0 y +100 kHz para 1, 
# el umbral de corte es perfectamente el CERO absoluto (0 Hz).
bits_rx = []
for i in range(len(bits_tx)):
    inicio_b = int((i + 0.25) * muestras_por_bit)
    fin_b = int((i + 0.75) * muestras_por_bit)
    
    freq_media_bit = np.mean(freq_sincronizada[inicio_b:fin_b])
    
    if freq_media_bit > 0.0:
        bits_rx.append(1)
    else:
        bits_rx.append(0)

bits_rx = np.array(bits_rx)

# ==========================================
# 5. CÁLCULO DE BER (Fase D)
# ==========================================
errores = np.sum(bits_tx != bits_rx)
ber = errores / len(bits_tx)

print("\n" + "="*50)
print(" RESULTADOS DE DEMODULACIÓN FSK (Fase C & D)")
print("="*50)
print(f"  Total de bits evaluados (PN-9):  {len(bits_tx)} bits")
print(f"  Bits erróneos recibidos:         {errores}")
print(f"  TASA DE ERROR DE BIT (BER):      {ber:.6f}")
if ber < 1e-3:
    print("  ESTADO: ¡CUMPLE ESPECIFICACIÓN IEEE! (BER < 1e-3)")
else:
    print("  ESTADO: Revisa la orientación de tus antenas.")
print("="*50 + "\n")

# ==========================================
# 6. GRÁFICAS DE COMPROBACIÓN
# ==========================================
bits_ver = 20 
muestras_ver = bits_ver * muestras_por_bit
tiempo = np.arange(muestras_ver) / sample_rate * 1000

plt.figure(figsize=(12, 8))

# Subgráfica 1: Frecuencia instantánea detectada en el aire
plt.subplot(2, 1, 1)
plt.plot(tiempo, freq_sincronizada[:muestras_ver] / 1e3, 'c-', linewidth=2, label='Frecuencia RX Detectada (kHz)')
plt.axhline(100.0, color='blue', linestyle=':', label='Frecuencia Ideal Bit 1 (+100 kHz)')
plt.axhline(-100.0, color='red', linestyle=':', label='Frecuencia Ideal Bit 0 (-100 kHz)')
plt.axhline(0.0, color='green', linestyle='--', linewidth=2, label='Umbral de Decisión (0 Hz)')
plt.title(f'Demodulación FSK - Discriminador de Fase | BER Experimental = {ber:.4f}')
plt.ylabel('Desviación de Frecuencia (kHz)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.ylim([-150, 150])

# Subgráfica 2: Comparativa de bits transmitidos vs recibidos
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

# Liberación obligatoria de memoria
sdr.tx_destroy_buffer()
print("Memoria del ADALM-Pluto liberada correctamente.")