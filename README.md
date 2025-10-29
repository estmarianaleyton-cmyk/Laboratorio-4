# Laboratorio 4 - Señales electromiográficas (EMG)

**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 26 de Octubre del 2025

**Título de la práctica:** Señales electromiográficas (EMG)

# **Objetivos**

- Aplicar el filtrado de señales continuas para procesar una señal electromiográfica (EMG).
- Detectar la aparición de fatiga muscular mediante el análisis espectral de contracciones musculares individuales.
- Comparar el comportamiento de una señal emulada y una señal real en términos de frecuencia media y mediana.
- Emplear herramientas computacionales para el procesamiento, segmentación y análisis de señales biomédicas.

# **Parte A**

## **Código en Python (Google colab)**
<pre> ```
# Importación de las librerias a utilizar
from google.colab import files
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
  
# Cargar los archivos desde el computador
uploaded = files.upload()
voltaje = np.loadtxt("EMG_señal.txt")
  
# Definir parámetros de muestreo
fs = 1500
N = len(voltaje)
t = np.arange(N) / fs  # eje de tiempo

# Graficar la señal sin filtar
plt.figure(figsize=(10,4))
plt.plot(t, voltaje, label="Señal EOG, tomada en el laboratorio")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title("Señal EMG simulada")
plt.grid(True)
plt.show()

# Sacar la media de la señal para eliminar el offset
voltaje_sin_offset = voltaje - np.mean(voltaje)
  
# Graficar señal sin offset
plt.figure(figsize=(10,4))
plt.plot(t, voltaje_sin_offset, label="Señal EMG sin offset")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title("Señal EMG filtrada (sin offset)")
plt.grid(True)
plt.legend()
plt.show()
```
</pre>
## **Diagrama de flujo**

## **Gráfica de la señal EMG simulada**
<img width="1073" height="494" alt="image" src="https://github.com/user-attachments/assets/1479fac0-cc57-4f4e-b37c-b8864b5b2397" />

## **Gráfica de la señal EMG simulada filtrada**
<img width="1068" height="493" alt="image" src="https://github.com/user-attachments/assets/3808bbcc-fb43-464f-82c5-b99c9b622088" />

## **Segmentación de la señal**

## **Código en Python (Google colab)**
<pre> ```
# Dividir en 5 segmentos iguales, lo que corresponderia 5 contracciones simuladas
n = len(voltaje)
segmentos = np.array_split(voltaje, 5)

# Graficar cada contracción segmentada
plt.figure(figsize=(10,6))
for i, seg in enumerate(segmentos):
    plt.plot(seg + i*0.5, label=f'Contracción {i+1}')  # se separan para verlas mejor
plt.title("Segmentación de las 5 contracciones simuladas")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (V) + desplazamiento")
plt.legend()
plt.show()
  ```
</pre>
## **Diagrama de flujo**

## **Señal segmentada (5 contracciones simuladas)**
<img width="963" height="613" alt="image" src="https://github.com/user-attachments/assets/16a4eb6c-d626-49ef-8757-5955661974bb" />

## **Frecuencia media y mediana de cada segmento**

## **Código en Python (Google colab)**
<pre> ```
def filtrar_pasaaltos(x, fs, fc=20):
b, a = butter(4, fc/(fs/2), btype='high')
return filtfilt(b, a, x)

# Función para calcular frecuencia media y mediana de un segmento
def calcular_frecuencias(segmento, fs):
segmento = segmento - np.mean(segmento)
segmento = segmento / np.max(np.abs(segmento))

N = len(segmento)
freqs = fftfreq(N, 1/fs)
espectro = np.abs(fft(segmento))**2

# Tomar solo frecuencias positivas
freqs = freqs[:N//2]
espectro = espectro[:N//2]

# Filtrar frecuencias bajas (<20 Hz)
filtro = freqs > 20
freqs = freqs[filtro]
espectro = espectro[filtro]

# Frecuencia media
f_media = np.sum(freqs * espectro) / np.sum(espectro)

# Frecuencia mediana
potencia_acumulada = np.cumsum(espectro)
mitad = potencia_acumulada[-1] / 2
idx = np.where(potencia_acumulada >= mitad)[0][0]
f_mediana = freqs[idx]
segmento = filtrar_pasaaltos(segmento, fs)

return f_media, f_mediana

# Calcular frecuencias para cada segmento
resultados = []
for i, seg in enumerate(segmentos, start=1):
f_media, f_mediana = calcular_frecuencias(seg, fs)
resultados.append({
"Contracción": i,
"Frecuencia Media (Hz)": f_media,
"Frecuencia Mediana (Hz)": f_mediana
})

# Crear tabla con los resultados
tabla = pd.DataFrame(resultados)
print("\nResultados de frecuencias por contracción:\n")
print(tabla.to_string(index=False))

# Gráficos de evolución de las frecuencias
plt.figure(figsize=(8,5))
plt.plot(tabla["Contracción"], tabla["Frecuencia Media (Hz)"], 'o-r', label="Frecuencia Media")
plt.plot(tabla["Contracción"], tabla["Frecuencia Mediana (Hz)"], 's-g', label="Frecuencia Mediana")
plt.title("Evolución de las frecuencias por contracción")
plt.xlabel("Número de contracción")
plt.ylabel("Frecuencia (Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
  ```
</pre>
## **Diagrama de flujo**

## **Resultados de frecuencias por contración:**
<img width="519" height="121" alt="image" src="https://github.com/user-attachments/assets/7163de94-881d-41a0-b8f8-8f43a9802630" />

## **Gráfica de la evolución de las frecuencias por contracción**
<img width="887" height="550" alt="image" src="https://github.com/user-attachments/assets/60f6c1ad-4de4-4d3d-9700-6e877bcfe5e7" />
