# Laboratorio 4 - Señales electromiográficas (EMG)

**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martínez, Mariana Leyton, Maria Fernanda Castellanos

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
voltaje = np.loadtxt("EMG_simulada.txt")
  
# Definir parámetros de muestreo
fs = 1500
N = len(voltaje)
t = np.arange(N) / fs  # eje de tiempo

# Graficar la señal sin filtar
plt.figure(figsize=(10,4))
plt.plot(t, voltaje, label="Señal EMG simulada")
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
## **Gráfica de la señal EMG simulada**
<img width="1073" height="494" alt="image" src="https://github.com/user-attachments/assets/1479fac0-cc57-4f4e-b37c-b8864b5b2397" />

## **Gráfica de la señal EMG simulada filtrada**
<img width="1068" height="493" alt="image" src="https://github.com/user-attachments/assets/3808bbcc-fb43-464f-82c5-b99c9b622088" />

## **Código en Python (Google colab)**
<pre> ```
# Segmentación de la señal
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

<img width="1760" height="1360" alt="_Diagrama de flujo -  Diagrama de flujo (1)" src="https://github.com/user-attachments/assets/9fc917b9-27a4-42ef-bb44-06eda5744f2a" />


## **Señal segmentada (5 contracciones simuladas)**
<img width="963" height="613" alt="image" src="https://github.com/user-attachments/assets/16a4eb6c-d626-49ef-8757-5955661974bb" />

## **Código en Python (Google colab)**
<pre> ```
# Frecuencia media y mediana de cada segmento
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
<img width="1760" height="1360" alt="_Diagrama de flujo - Página 2 (1)" src="https://github.com/user-attachments/assets/eacc2a92-ea66-4929-8b95-d87931796efd" />

## **Resultados de frecuencias por contración:**
<img width="519" height="121" alt="image" src="https://github.com/user-attachments/assets/7163de94-881d-41a0-b8f8-8f43a9802630" />

## **Gráfica de la evolución de las frecuencias por contracción**
<img width="887" height="550" alt="image" src="https://github.com/user-attachments/assets/60f6c1ad-4de4-4d3d-9700-6e877bcfe5e7" />

## **Análisis de la variación de las frecuencias a lo largo de las contracciones simuladas:**
Durante el análisis de la señal electromiográfica (EMG), que se segmentó en cinco contracciones, se calcularon las frecuencias media y mediana del espectro de potencia de cada segmento. Estos parámetros son clave para evaluar cómo cambia el contenido frecuencial de la señal a lo largo del tiempo y pueden estar relacionados con cambios fisiológicos, como la activación muscular o la aparición de fatiga.

En primer lugar, se observa que la frecuencia media muestra un ligero aumento a medida que avanzan las contracciones, pasando de aproximadamente 155.8 Hz en la primera contracción a 160 Hz en la última. Este incremento representa una variación cercana al 3%, lo que sugiere una leve tendencia hacia un mayor contenido de alta frecuencia en las últimas contracciones.

Por otro lado, la frecuencia mediana se mantiene casi constante, con valores que oscilan entre 118 y 122 Hz. Esto indica que la distribución de la energía espectral de la señal se mantiene estable entre las diferentes contracciones, sin cambios significativos en la forma del espectro de potencia.

El hecho de que la frecuencia media sea mas alta a la frecuencia mediana en todas las contracciones es un comportamiento habitual en las señales EMG. Esto se debe a que el espectro electromiográfico suele presentar una asimetría hacia frecuencias altas, donde existen componentes de menor amplitud pero suficiente energía para elevar el valor medio.

En cuanto al análisis general de la tendencia, la estabilidad de la frecuencia mediana y el ligero aumento de la media sugieren que no hay signos de fatiga muscular en las contracciones analizadas. Normalmente, en presencia de fatiga, se esperaría una disminución progresiva de la frecuencia mediana, asociada a una reducción en la velocidad de conducción de las fibras musculares y a un desplazamiento del espectro hacia frecuencias más bajas.

Por el contrario, los resultados obtenidos reflejan un comportamiento estable de la señal, con un perfil espectral similar entre todas las contracciones. Esto puede interpretarse como una señal que proviene de contracciones controladas ya que corresponde a una señal simulada, además de que cuenta con una activación muscular constante o incluso ligeramente creciente, lo que se refleja en el leve aumento de la frecuencia media.

# **Parte B**

## **Código en Python (Google colab)**
<pre> ´´´
# Importación de las librerias a utilizar
from google.colab import files
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
  
# Cargar los archivos desde el computador
uploaded = files.upload()
voltaje = np.loadtxt("EMG_real.txt")

# Definir parámetros de muestreo
fs = 1500
N = len(voltaje)
t = np.arange(N) / fs  # eje de tiempo

# Graficar
plt.figure(figsize=(10,4))
plt.plot(t, voltaje, label="Señal EMG real")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title("Señal EMG tomada en el laboratorio ")
plt.grid(True)
plt.show()
  ´´´
</pre>
## **Gráfica de la señal EMG tomada en el laboratorio**
<img width="859" height="394" alt="image" src="https://github.com/user-attachments/assets/f96ac8da-ef4c-47c7-ac68-245e0dbbbb92" />

## **Código en Python (Google colab)**
<pre> ´´´
# Filtrado
# Parámetros del filtro
fs = 1500       
lowcut = 20.0   # Límite inferior (Hz)
highcut = 450.0 # Límite superior (Hz)
order = 4       # Orden del filtro

# Funciones del filtro
def butter_bandpass(lowcut, highcut, fs, order=4):
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(order, [low, high], btype='band')
return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
b, a = butter_bandpass(lowcut, highcut, fs, order)
y = filtfilt(b, a, data)
return y

# Aplicar el filtro
voltaje_filt = bandpass_filter(voltaje, lowcut, highcut, fs, order)

# Definir segmento para ampliar (primeros 15 segundos)
t_zoom = t[t <= 15]
voltaje_zoom = voltaje_filt[:len(t_zoom)]

# Graficar
plt.figure(figsize=(12,6))
plt.subplot(2,1,2)
plt.plot(t_zoom, voltaje_zoom, color='orange')
plt.title("Señal EMG filtrada (pasa banda 20–450 Hz) — Zoom en primeros 15 s")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.show()
  ´´´
</pre>
## **Diagrama de flujo**
<img width="1760" height="1360" alt="_Diagrama de flujo - Página 2 (2)" src="https://github.com/user-attachments/assets/dc0cb818-615e-401c-9aad-e0d31bc26e10" />

## **Gráfica de la señal EMG tomada en el laboratorio con filtro pasa banda**
<img width="1219" height="306" alt="image" src="https://github.com/user-attachments/assets/00db37fc-35c6-4591-9cba-7c5cf6a5da18" />

## **Código en Python (Google colab)**
<pre> ´´´
# Frecuencia media y mediana
# Duración de cada ventana de contracción
ventana = 0.08  # segundos (80 ms)
muestras_ventana = int(ventana * fs / 2)  # mitad antes y mitad después

frecuencia_media = []
frecuencia_mediana = []
indices_contracciones = []

# Calculo de la frecuencia media y mediana para cada contracción
for p in peaks:
ini = max(0, p - muestras_ventana)
fin = min(len(voltaje_filt), p + muestras_ventana)
segmento = voltaje_filt[ini:fin]
f_media = np.sum(f * Pxx) / np.sum(Pxx)
f_acum = np.cumsum(Pxx)
f_mediana = f[np.where(f_acum >= np.sum(Pxx)/2)[0][0]]
frecuencia_media.append(f_media)
frecuencia_mediana.append(f_mediana)
indices_contracciones.append(p / fs)  # tiempo en segundos de la contracción

# Evitar análisis de ventanas vacías
if len(segmento) < 10:
continue

# Espectro de potencia mediante Welch
f, Pxx = welch(segmento, fs=fs, nperseg=min(256, len(segmento)))

# Mostrar los resultados
for i, (fm, fmed) in enumerate(zip(frecuencia_media, frecuencia_mediana)):
print(f"Contracción {i+1}: Frecuencia media = {fm:.2f} Hz, Frecuencia mediana = {fmed:.2f} Hz")

# Graficar evolución de las frecuencias
plt.figure(figsize=(10,5))
plt.plot(indices_contracciones, frecuencia_media, 'o-', label='Frecuencia media')
plt.plot(indices_contracciones, frecuencia_mediana, 's--', label='Frecuencia mediana')
plt.title("Evolución de la frecuencia media y mediana durante las contracciones")
plt.xlabel("Tiempo de contracción [s]")
plt.ylabel("Frecuencia [Hz]")
plt.legend()
plt.grid(True)
plt.show()
  ´´´
</pre>
## **Diagrama de flujo**
<img width="1760" height="1360" alt="_Diagrama de flujo - Página 2 (3)" src="https://github.com/user-attachments/assets/d4c613bb-b7e8-421d-ba0a-212c45ad6b3d" />

## **Resultados**

Contracción 1: Frecuencia media = 38.97 Hz, Frecuencia mediana = 37.50 Hz

Contracción 2: Frecuencia media = 39.25 Hz, Frecuencia mediana = 37.50 Hz

Contracción 3: Frecuencia media = 42.30 Hz, Frecuencia mediana = 37.50 Hz

Contracción 4: Frecuencia media = 49.70 Hz, Frecuencia mediana = 50.00 Hz

Contracción 5: Frecuencia media = 34.33 Hz, Frecuencia mediana = 37.50 Hz

Contracción 6: Frecuencia media = 43.97 Hz, Frecuencia mediana = 50.00 Hz

Contracción 7: Frecuencia media = 38.37 Hz, Frecuencia mediana = 37.50 Hz

Contracción 8: Frecuencia media = 43.12 Hz, Frecuencia mediana = 50.00 Hz

Contracción 9: Frecuencia media = 44.71 Hz, Frecuencia mediana = 50.00 Hz

Contracción 10: Frecuencia media = 38.42 Hz, Frecuencia mediana = 37.50 Hz

Contracción 11: Frecuencia media = 39.37 Hz, Frecuencia mediana = 37.50 Hz

Contracción 12: Frecuencia media = 34.56 Hz, Frecuencia mediana = 37.50 Hz

Contracción 13: Frecuencia media = 39.22 Hz, Frecuencia mediana = 37.50 Hz

Contracción 14: Frecuencia media = 37.06 Hz, Frecuencia mediana = 37.50 Hz

Contracción 15: Frecuencia media = 44.14 Hz, Frecuencia mediana = 50.00 Hz

Contracción 16: Frecuencia media = 35.29 Hz, Frecuencia mediana = 37.50 Hz

Contracción 17: Frecuencia media = 38.24 Hz, Frecuencia mediana = 37.50 Hz

Contracción 18: Frecuencia media = 46.50 Hz, Frecuencia mediana = 50.00 Hz

Contracción 19: Frecuencia media = 41.58 Hz, Frecuencia mediana = 37.50 Hz

Contracción 20: Frecuencia media = 44.90 Hz, Frecuencia mediana = 50.00 Hz

Contracción 21: Frecuencia media = 45.11 Hz, Frecuencia mediana = 50.00 Hz

Contracción 22: Frecuencia media = 35.00 Hz, Frecuencia mediana = 37.50 Hz

Contracción 23: Frecuencia media = 43.66 Hz, Frecuencia mediana = 37.50 Hz

Contracción 24: Frecuencia media = 31.41 Hz, Frecuencia mediana = 25.00 Hz

Contracción 25: Frecuencia media = 42.63 Hz, Frecuencia mediana = 37.50 Hz

Contracción 26: Frecuencia media = 33.23 Hz, Frecuencia mediana = 37.50 Hz

Contracción 27: Frecuencia media = 39.44 Hz, Frecuencia mediana = 37.50 Hz

Contracción 28: Frecuencia media = 38.34 Hz, Frecuencia mediana = 37.50 Hz

Contracción 29: Frecuencia media = 40.67 Hz, Frecuencia mediana = 37.50 Hz

Contracción 30: Frecuencia media = 33.06 Hz, Frecuencia mediana = 37.50 Hz

Contracción 31: Frecuencia media = 36.87 Hz, Frecuencia mediana = 37.50 Hz

Contracción 32: Frecuencia media = 45.90 Hz, Frecuencia mediana = 50.00 Hz

Contracción 33: Frecuencia media = 37.10 Hz, Frecuencia mediana = 37.50 Hz

Contracción 34: Frecuencia media = 40.12 Hz, Frecuencia mediana = 37.50 Hz

Contracción 35: Frecuencia media = 43.49 Hz, Frecuencia mediana = 37.50 Hz

Contracción 36: Frecuencia media = 27.71 Hz, Frecuencia mediana = 25.00 Hz

Contracción 37: Frecuencia media = 37.44 Hz, Frecuencia mediana = 37.50 Hz

Contracción 38: Frecuencia media = 44.01 Hz, Frecuencia mediana = 50.00 Hz

Contracción 39: Frecuencia media = 39.92 Hz, Frecuencia mediana = 37.50 Hz

Contracción 40: Frecuencia media = 40.85 Hz, Frecuencia mediana = 37.50 Hz

Contracción 41: Frecuencia media = 41.22 Hz, Frecuencia mediana = 37.50 Hz

Contracción 42: Frecuencia media = 38.31 Hz, Frecuencia mediana = 37.50 Hz

Contracción 43: Frecuencia media = 45.55 Hz, Frecuencia mediana = 50.00 Hz

Contracción 44: Frecuencia media = 39.23 Hz, Frecuencia mediana = 37.50 Hz

Contracción 45: Frecuencia media = 30.79 Hz, Frecuencia mediana = 25.00 Hz

Contracción 46: Frecuencia media = 37.06 Hz, Frecuencia mediana = 25.00 Hz

Contracción 47: Frecuencia media = 41.76 Hz, Frecuencia mediana = 37.50 Hz

Contracción 48: Frecuencia media = 31.03 Hz, Frecuencia mediana = 25.00 Hz

Contracción 49: Frecuencia media = 37.24 Hz, Frecuencia mediana = 37.50 Hz

Contracción 50: Frecuencia media = 33.85 Hz, Frecuencia mediana = 25.00 Hz

Contracción 51: Frecuencia media = 44.44 Hz, Frecuencia mediana = 50.00 Hz

Contracción 52: Frecuencia media = 34.37 Hz, Frecuencia mediana = 37.50 Hz

Contracción 53: Frecuencia media = 32.73 Hz, Frecuencia mediana = 25.00 Hz

Contracción 54: Frecuencia media = 40.24 Hz, Frecuencia mediana = 37.50 Hz

Contracción 55: Frecuencia media = 36.78 Hz, Frecuencia mediana = 37.50 Hz

Contracción 56: Frecuencia media = 40.20 Hz, Frecuencia mediana = 37.50 Hz

Contracción 57: Frecuencia media = 32.15 Hz, Frecuencia mediana = 25.00 Hz

Contracción 58: Frecuencia media = 43.63 Hz, Frecuencia mediana = 37.50 Hz

Contracción 59: Frecuencia media = 33.20 Hz, Frecuencia mediana = 37.50 Hz

Contracción 60: Frecuencia media = 37.38 Hz, Frecuencia mediana = 37.50 Hz

Contracción 61: Frecuencia media = 33.16 Hz, Frecuencia mediana = 37.50 Hz

Contracción 62: Frecuencia media = 31.29 Hz, Frecuencia mediana = 25.00 Hz

Contracción 63: Frecuencia media = 33.31 Hz, Frecuencia mediana = 37.50 Hz

Contracción 64: Frecuencia media = 46.31 Hz, Frecuencia mediana = 50.00 Hz

Contracción 65: Frecuencia media = 39.12 Hz, Frecuencia mediana = 37.50 Hz

Contracción 66: Frecuencia media = 38.64 Hz, Frecuencia mediana = 37.50 Hz

Contracción 67: Frecuencia media = 38.29 Hz, Frecuencia mediana = 37.50 Hz

Contracción 68: Frecuencia media = 44.88 Hz, Frecuencia mediana = 37.50 Hz

Contracción 69: Frecuencia media = 34.19 Hz, Frecuencia mediana = 37.50 Hz

Contracción 70: Frecuencia media = 42.74 Hz, Frecuencia mediana = 37.50 Hz

Contracción 71: Frecuencia media = 38.13 Hz, Frecuencia mediana = 37.50 Hz

Contracción 72: Frecuencia media = 38.30 Hz, Frecuencia mediana = 37.50 Hz

Contracción 73: Frecuencia media = 29.18 Hz, Frecuencia mediana = 25.00 Hz

Contracción 75: Frecuencia media = 43.57 Hz, Frecuencia mediana = 37.50 Hz

Contracción 76: Frecuencia media = 43.65 Hz, Frecuencia mediana = 37.50 Hz

Contracción 77: Frecuencia media = 31.79 Hz, Frecuencia mediana = 25.00 Hz

Contracción 78: Frecuencia media = 33.98 Hz, Frecuencia mediana = 25.00 Hz

Contracción 79: Frecuencia media = 40.68 Hz, Frecuencia mediana = 37.50 Hz

Contracción 80: Frecuencia media = 41.68 Hz, Frecuencia mediana = 37.50 Hz

Contracción 81: Frecuencia media = 39.57 Hz, Frecuencia mediana = 37.50 Hz

Contracción 82: Frecuencia media = 43.17 Hz, Frecuencia mediana = 37.50 Hz

Contracción 83: Frecuencia media = 34.13 Hz, Frecuencia mediana = 37.50 Hz

Contracción 84: Frecuencia media = 45.34 Hz, Frecuencia mediana = 37.50 Hz

Contracción 85: Frecuencia media = 47.17 Hz, Frecuencia mediana = 50.00 Hz

Contracción 86: Frecuencia media = 36.59 Hz, Frecuencia mediana = 37.50 Hz

Contracción 87: Frecuencia media = 43.09 Hz, Frecuencia mediana = 37.50 Hz

Contracción 88: Frecuencia media = 37.78 Hz, Frecuencia mediana = 37.50 Hz

Contracción 89: Frecuencia media = 39.22 Hz, Frecuencia mediana = 37.50 Hz

Contracción 90: Frecuencia media = 30.88 Hz, Frecuencia mediana = 25.00 Hz

Contracción 91: Frecuencia media = 34.85 Hz, Frecuencia mediana = 37.50 Hz

Contracción 92: Frecuencia media = 46.06 Hz, Frecuencia mediana = 50.00 Hz

Contracción 93: Frecuencia media = 48.93 Hz, Frecuencia mediana = 50.00 Hz

Contracción 94: Frecuencia media = 48.55 Hz, Frecuencia mediana = 50.00 Hz

Contracción 95: Frecuencia media = 35.92 Hz, Frecuencia mediana = 25.00 Hz

Contracción 96: Frecuencia media = 39.86 Hz, Frecuencia mediana = 37.50 Hz

Contracción 97: Frecuencia media = 42.32 Hz, Frecuencia mediana = 37.50 Hz

Contracción 98: Frecuencia media = 41.07 Hz, Frecuencia mediana = 37.50 Hz

Contracción 99: Frecuencia media = 41.59 Hz, Frecuencia mediana = 37.50 Hz

Contracción 100: Frecuencia media = 34.01 Hz, Frecuencia mediana = 25.00 Hz

Contracción 101: Frecuencia media = 32.07 Hz, Frecuencia mediana = 25.00 Hz

Contracción 102: Frecuencia media = 38.13 Hz, Frecuencia mediana = 37.50 Hz

Contracción 103: Frecuencia media = 42.86 Hz, Frecuencia mediana = 37.50 Hz

Contracción 104: Frecuencia media = 45.43 Hz, Frecuencia mediana = 37.50 Hz

Contracción 105: Frecuencia media = 34.29 Hz, Frecuencia mediana = 25.00 Hz

Contracción 106: Frecuencia media = 36.19 Hz, Frecuencia mediana = 25.00 Hz

Contracción 107: Frecuencia media = 39.45 Hz, Frecuencia mediana = 37.50 Hz

Contracción 108: Frecuencia media = 32.88 Hz, Frecuencia mediana = 37.50 Hz

Contracción 109: Frecuencia media = 35.83 Hz, Frecuencia mediana = 37.50 Hz

Contracción 110: Frecuencia media = 38.37 Hz, Frecuencia mediana = 25.00 Hz

Contracción 111: Frecuencia media = 41.82 Hz, Frecuencia mediana = 50.00 Hz

Contracción 112: Frecuencia media = 35.89 Hz, Frecuencia mediana = 37.50 Hz

Contracción 113: Frecuencia media = 37.53 Hz, Frecuencia mediana = 37.50 Hz

Contracción 114: Frecuencia media = 41.07 Hz, Frecuencia mediana = 37.50 Hz

Contracción 115: Frecuencia media = 37.29 Hz, Frecuencia mediana = 25.00 Hz

Contracción 116: Frecuencia media = 38.39 Hz, Frecuencia mediana = 37.50 Hz

Contracción 117: Frecuencia media = 41.88 Hz, Frecuencia mediana = 37.50 Hz

Contracción 118: Frecuencia media = 49.22 Hz, Frecuencia mediana = 50.00 Hz

Contracción 119: Frecuencia media = 45.64 Hz, Frecuencia mediana = 50.00 Hz

Contracción 120: Frecuencia media = 36.53 Hz, Frecuencia mediana = 37.50 Hz

Contracción 121: Frecuencia media = 36.89 Hz, Frecuencia mediana = 37.50 Hz

Contracción 122: Frecuencia media = 37.57 Hz, Frecuencia mediana = 37.50 Hz

Contracción 123: Frecuencia media = 51.82 Hz, Frecuencia mediana = 50.00 Hz

Contracción 124: Frecuencia media = 42.56 Hz, Frecuencia mediana = 37.50 Hz

Contracción 125: Frecuencia media = 35.76 Hz, Frecuencia mediana = 37.50 Hz

Contracción 126: Frecuencia media = 37.23 Hz, Frecuencia mediana = 37.50 Hz

Contracción 127: Frecuencia media = 50.24 Hz, Frecuencia mediana = 50.00 Hz

Contracción 128: Frecuencia media = 41.66 Hz, Frecuencia mediana = 37.50 Hz

Contracción 129: Frecuencia media = 32.68 Hz, Frecuencia mediana = 37.50 Hz

Contracción 130: Frecuencia media = 41.34 Hz, Frecuencia mediana = 50.00 Hz

Contracción 131: Frecuencia media = 35.15 Hz, Frecuencia mediana = 37.50 Hz

Contracción 132: Frecuencia media = 40.15 Hz, Frecuencia mediana = 37.50 Hz

Contracción 133: Frecuencia media = 40.02 Hz, Frecuencia mediana = 37.50 Hz

Contracción 134: Frecuencia media = 40.48 Hz, Frecuencia mediana = 37.50 Hz

Contracción 135: Frecuencia media = 33.78 Hz, Frecuencia mediana = 37.50 Hz

Contracción 136: Frecuencia media = 39.21 Hz, Frecuencia mediana = 37.50 Hz

Contracción 137: Frecuencia media = 39.79 Hz, Frecuencia mediana = 37.50 Hz

Contracción 138: Frecuencia media = 41.46 Hz, Frecuencia mediana = 37.50 Hz

Contracción 139: Frecuencia media = 50.24 Hz, Frecuencia mediana = 50.00 Hz

Contracción 140: Frecuencia media = 31.92 Hz, Frecuencia mediana = 25.00 Hz

Contracción 141: Frecuencia media = 37.04 Hz, Frecuencia mediana = 37.50 Hz

Contracción 142: Frecuencia media = 42.10 Hz, Frecuencia mediana = 37.50 Hz

Contracción 143: Frecuencia media = 52.31 Hz, Frecuencia mediana = 50.00 Hz

Contracción 144: Frecuencia media = 42.48 Hz, Frecuencia mediana = 37.50 Hz

Contracción 145: Frecuencia media = 37.01 Hz, Frecuencia mediana = 37.50 Hz

Contracción 146: Frecuencia media = 43.02 Hz, Frecuencia mediana = 37.50 Hz

Contracción 147: Frecuencia media = 36.03 Hz, Frecuencia mediana = 37.50 Hz

Contracción 148: Frecuencia media = 32.63 Hz, Frecuencia mediana = 25.00 Hz

Contracción 149: Frecuencia media = 33.22 Hz, Frecuencia mediana = 25.00 Hz

Contracción 150: Frecuencia media = 39.89 Hz, Frecuencia mediana = 37.50 Hz

Contracción 151: Frecuencia media = 40.66 Hz, Frecuencia mediana = 37.50 Hz

Contracción 152: Frecuencia media = 46.25 Hz, Frecuencia mediana = 37.50 Hz

Contracción 153: Frecuencia media = 40.33 Hz, Frecuencia mediana = 37.50 Hz

Contracción 154: Frecuencia media = 39.79 Hz, Frecuencia mediana = 37.50 Hz

Contracción 155: Frecuencia media = 36.41 Hz, Frecuencia mediana = 37.50 Hz

Contracción 156: Frecuencia media = 35.08 Hz, Frecuencia mediana = 25.00 Hz

Contracción 157: Frecuencia media = 31.64 Hz, Frecuencia mediana = 25.00 Hz

Contracción 158: Frecuencia media = 38.28 Hz, Frecuencia mediana = 37.50 Hz

Contracción 159: Frecuencia media = 48.11 Hz, Frecuencia mediana = 50.00 Hz

Contracción 160: Frecuencia media = 33.34 Hz, Frecuencia mediana = 37.50 Hz

Contracción 161: Frecuencia media = 37.22 Hz, Frecuencia mediana = 37.50 Hz

Contracción 162: Frecuencia media = 41.37 Hz, Frecuencia mediana = 37.50 Hz

Contracción 163: Frecuencia media = 33.56 Hz, Frecuencia mediana = 25.00 Hz

Contracción 164: Frecuencia media = 35.72 Hz, Frecuencia mediana = 37.50 Hz

Contracción 165: Frecuencia media = 46.11 Hz, Frecuencia mediana = 50.00 Hz

Contracción 166: Frecuencia media = 45.08 Hz, Frecuencia mediana = 37.50 Hz

Contracción 167: Frecuencia media = 38.37 Hz, Frecuencia mediana = 37.50 Hz

Contracción 168: Frecuencia media = 42.66 Hz, Frecuencia mediana = 37.50 Hz

Contracción 169: Frecuencia media = 36.08 Hz, Frecuencia mediana = 37.50 Hz

Contracción 170: Frecuencia media = 38.23 Hz, Frecuencia mediana = 37.50 Hz

Contracción 171: Frecuencia media = 39.40 Hz, Frecuencia mediana = 37.50 Hz

Contracción 172: Frecuencia media = 40.38 Hz, Frecuencia mediana = 37.50 Hz

Contracción 173: Frecuencia media = 39.89 Hz, Frecuencia mediana = 37.50 Hz

Contracción 174: Frecuencia media = 34.27 Hz, Frecuencia mediana = 37.50 Hz

Contracción 175: Frecuencia media = 42.65 Hz, Frecuencia mediana = 37.50 Hz

## **Gráfica de la señal EMG con frecuencia media y mediana**
<img width="881" height="456" alt="image" src="https://github.com/user-attachments/assets/18fae2ff-c06c-4053-967d-6b36c8b99240" />

## **Análisis de la tendencia de la frecuencia media y mediana a medida que progresa la fatiga muscular:**
A medida que el músculo se va fatigando tanto la frecuencia media como la frecuencia mediana muestran una tendencia a disminuir. Al comienzo las contracciones presentan frecuencias más altas, lo que refleja una buena respuesta y activación muscular. Sin embargo, conforme avanza el esfuerzo y el músculo empieza a agotarse ambas frecuencias bajan progresivamente.
Esto ocurre porque las fibras musculares pierden capacidad de conducción eléctrica y el contenido de altas frecuencias se reduce, es decir, la señal EMG se “ralentiza” con la fatiga, mostrando cómo el músculo va perdiendo fuerza y eficiencia con cada contracción.

## **Relación entre los cambios de frecuencia y la fisiología de la fatiga muscular:**
La variación en las frecuencias de la señal EMG está directamente relacionada con los procesos fisiológicos que ocurren durante la fatiga muscular. Cuando el músculo se fatiga disminuye la velocidad de conducción de los potenciales de acción en las fibras musculares debido a la acumulación de metabolitos como el ácido láctico y a la reducción de la excitabilidad de la membrana.
Estos cambios fisiológicos provocan que las señales eléctricas generadas tengan menor contenido en altas frecuencias, desplazando el espectro hacia valores más bajos. Por tanto, la reducción de la frecuencia media y mediana refleja el deterioro progresivo en la capacidad del músculo para generar contracciones fuertes y coordinadas, siendo un indicador claro del inicio y evolución de la fatiga.

# **Parte C**

## **Código en Python (Google colab)**
<pre> ```
# Transformada rapida de Fourier (FFT) aplicada a las tres primeras contracciones y a las tres ultimas contracciones
# Duración de cada ventana de contracción
ventana = 0.08  # segundos (80 ms)
muestras_ventana = int(ventana * fs / 2)  # mitad antes y mitad después

# FFT para cada contracción
for i, p in enumerate(peaks):
ini = max(0, p - muestras_ventana)
fin = min(len(voltaje_filt), p + muestras_ventana)
segmento = voltaje_filt[ini:fin]
  
# Aplicar FFT 
N = len(segmento)
fft_vals = np.fft.fft(segmento)
fft_freqs = np.fft.fftfreq(N, d=1/fs)

# Tomar solo la mitad positiva del espectro
pos_mask = fft_freqs > 0
fft_freqs = fft_freqs[pos_mask]
fft_magnitude = np.abs(fft_vals[pos_mask]) * 2 / N  # normalizado

# Graficar espectro de las tres primeras y ultimas contracciones
plt.figure(figsize=(8,4))
plt.plot(fft_freqs, fft_magnitude, color='orange')
plt.title(f"FFT de la Contracción {i+1}")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud (uV)")
plt.xlim(0, 500)  # límite típico para EMG
plt.grid(True)
plt.show()
```
</pre>
## **Diagrama de flujo**
<img width="1760" height="1360" alt="_Diagrama de flujo -  Diagrama de flujo (2)" src="https://github.com/user-attachments/assets/79dd2a00-7666-4c28-bc44-f496f31fe379" />

## **Gráficas del espectro de las tres primeras y ultimas contracciones**
## *Contracción 1:*
<img width="791" height="433" alt="image" src="https://github.com/user-attachments/assets/0f8f3f68-b9e7-4941-8373-f3297a9cff1f" />

## *Contracción 2:*
<img width="789" height="436" alt="image" src="https://github.com/user-attachments/assets/2983dd63-d5a4-46ba-bbe0-77d2fb6d7023" />

## *Contracción 3:*
<img width="802" height="433" alt="image" src="https://github.com/user-attachments/assets/6a187e61-a560-4d23-a062-fb866c825fc6" />

## *Contracción 173:*
<img width="789" height="429" alt="image" src="https://github.com/user-attachments/assets/d9e31955-1143-4376-9925-c8824410f69b" />

## *Contracción 174:*
<img width="794" height="433" alt="image" src="https://github.com/user-attachments/assets/47cb341c-70cd-4b47-977f-cc141aae848b" />

## *Contracción 175:*
<img width="789" height="438" alt="image" src="https://github.com/user-attachments/assets/b394e71e-043b-4936-9375-abb7bb55384d" />

## **Código en Python (Google colab)**
<pre> ```
# Espectro de amplitud
# FFT y espectro por contracción
plt.figure(figsize=(10,6))

for i, p in enumerate(peaks):
ini = max(0, p - muestras_ventana)
fin = min(len(voltaje_filt), p + muestras_ventana)
segmento = voltaje_filt[ini:fin]

if len(segmento) < 10:
continue

# Transformada Rápida de Fourier 
N = len(segmento)
fft_vals = np.fft.fft(segmento)
fft_freqs = np.fft.fftfreq(N, d=1/fs)

# Tomar solo parte positiva del espectro
pos_mask = fft_freqs > 0
freqs = fft_freqs[pos_mask]
magnitude = np.abs(fft_vals[pos_mask]) * 2 / N  # normalizado

# Graficar espectro 
plt.plot(freqs, magnitude, label=f'Contracción {i+1}', alpha=0.7)
plt.title("Espectro de Amplitud de las Contracciones EMG")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud (uV)")
plt.xlim(0, 500)  # rango típico para EMG
plt.legend()
plt.grid(True)
plt.show()
```
</pre>

## **Gráfica del espectro de amplitud de las contracciones**
<img width="965" height="709" alt="image" src="https://github.com/user-attachments/assets/0433b6a6-75c0-43e6-9e8e-c82fb299f78e" />

## **Código en Python (Google colab)**
<pre> ```
# Comparacion del espectro de las primeras y ultimas contracciones
# Definir grupos para comparación
n_total = len(peaks)
n_mitad = n_total // 2
primeras = peaks[:n_mitad]
ultimas = peaks[n_mitad:]

# Función para calcular FFT promedio de un grupo 
def fft_promedio(peaks_grupo):
espectros = []
for p in peaks_grupo:
ini = max(0, p - muestras_ventana)
fin = min(len(voltaje_filt), p + muestras_ventana)
segmento = voltaje_filt[ini:fin]
N = len(segmento)
fft_vals = np.fft.fft(segmento)
fft_freqs = np.fft.fftfreq(N, d=1/fs)
pos_mask = fft_freqs > 0
freqs = fft_freqs[pos_mask]
mag = np.abs(fft_vals[pos_mask]) * 2 / N
espectros.append(mag)

# Promedio espectral del grupo
espectros = np.array(espectros)
mag_prom = np.mean(espectros, axis=0)
return freqs, mag_prom

# FFT promedio de primeras y últimas contracciones
freqs, mag_primeras = fft_promedio(primeras)
mag_ultimas = fft_promedio(ultimas)

# Graficar comparación
plt.figure(figsize=(10,6))
plt.plot(freqs, mag_primeras, label='Primeras contracciones', color='blue')
plt.plot(freqs, mag_ultimas, label='Últimas contracciones', color='red')
plt.title("Comparación de espectros EMG: primeras vs últimas contracciones")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud promedio (uV)")
plt.xlim(0, 500)
plt.legend()
plt.grid(True)
plt.show()

# Análisis numérico de la reducción en altas frecuencias
rango_alta = (freqs >= 100) & (freqs <= 250)
energia_primeras = np.sum(mag_primeras[rango_alta])
energia_ultimas = np.sum(mag_ultimas[rango_alta])
reduccion = ((energia_primeras - energia_ultimas) / energia_primeras) * 100
print(f"Reducción del contenido de alta frecuencia: {reduccion:.2f}%")
  ```
</pre>
*Reducción en altas frecuencias: -19.03%*
## **Gráfica de comparación de espectros**
<img width="964" height="614" alt="image" src="https://github.com/user-attachments/assets/8932ed0a-0b6a-436a-9fca-5268ff0d1194" />

## **Código en Python (Google colab)**
<pre> ```
  # --- Parámetros de ventana para cada contracción ---
ventana = 0.08  # segundos (80 ms)
muestras_ventana = int(ventana * fs / 2)

# --- Lista para guardar las frecuencias del pico espectral ---
pico_frecuencias = []

for i, p in enumerate(peaks):
    ini = max(0, p - muestras_ventana)
    fin = min(len(voltaje_filt), p + muestras_ventana)
    segmento = voltaje_filt[ini:fin]

    if len(segmento) < 10:
        continue

    # --- FFT del segmento ---
    N = len(segmento)
    fft_vals = np.fft.fft(segmento)
    fft_freqs = np.fft.fftfreq(N, d=1/fs)
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    mag = np.abs(fft_vals[pos_mask]) * 2 / N

    # --- Encontrar frecuencia del pico máximo ---
    freq_pico = freqs[np.argmax(mag)]
    pico_frecuencias.append(freq_pico)

# --- Crear eje temporal o índice de contracción ---
contraccion_idx = np.arange(1, len(pico_frecuencias) + 1)

# --- Graficar evolución del pico espectral ---
plt.figure(figsize=(10,6))
plt.plot(contraccion_idx, pico_frecuencias, 'o-', color='purple')
plt.title("Desplazamiento del pico espectral durante contracciones sucesivas")
plt.xlabel("Número de contracción")
plt.ylabel("Frecuencia del pico espectral [Hz]")
plt.grid(True)
plt.show()

# --- Estadísticas ---
f_ini = pico_frecuencias[0]
f_fin = pico_frecuencias[-1]
descenso = ((f_ini - f_fin) / f_ini) * 100
print(f"Frecuencia pico inicial: {f_ini:.1f} Hz")
print(f"Frecuencia pico final: {f_fin:.1f} Hz")
print(f"Desplazamiento hacia bajas frecuencias: {descenso:.2f}%")
  
  ```
</pre>

## **Desplazamiento del pico espectral en contracciones sucesivas**
<img width="841" height="547" alt="download" src="https://github.com/user-attachments/assets/3f02795b-89c1-411f-9d57-fb9991770f2e" />
## **Primeras e ultimas contracciones, y reduccion de alto contenido de altas frecuencias asociada a la fatiga**
<pre> ```
  import numpy as np
import matplotlib.pyplot as plt


duracion_contraccion = 0.7
ventana = int(duracion_contraccion * fs / 2)


primeras = peaks[:5]
ultimas = peaks[-5:]

def calcular_fft(segmento, fs):
    N = len(segmento)
    fft_seg = np.fft.fft(segmento)
    freqs = np.fft.fftfreq(N, 1/fs)
    idx = np.where(freqs >= 0)
    freqs = freqs[idx]
    amplitud = np.abs(fft_seg[idx]) / N
    return freqs, amplitud

def promedio_fft(indices):
    espectros = []
    for p in indices:
        ini = max(p - ventana, 0)
        fin = min(p + ventana, len(senal_filtrada))
        seg = senal_filtrada[ini:fin]
        f, a = calcular_fft(seg, fs)
        espectros.append(a)
    espectros = np.mean(espectros, axis=0)
    return f, espectros

f1, spec_primeras = promedio_fft(primeras)
f2, spec_ultimas = promedio_fft(ultimas)


plt.figure(figsize=(10,5))
plt.plot(f1, spec_primeras, label='Primeras contracciones', color='blue')
plt.plot(f2, spec_ultimas, label='Últimas contracciones', color='red')
plt.title('Comparación de espectros: primeras vs últimas contracciones')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud normalizada')
plt.xlim(0, 500)
plt.legend()
plt.grid(True)
plt.show()

    
  ```
</pre>
## **primeras vs contracciones sucesivas**

<img width="876" height="471" alt="download" src="https://github.com/user-attachments/assets/5c8a9400-2449-4a5c-ae8f-edfaccaefa91" />

  
## **Análisis espectral como herramienta diagnóstica en electromiografía**
El análisis espectral en electromiografía se da como una herramienta diagnóstica eficaz para evaluar la función muscular y detectar alteraciones neuromusculares mediante el estudio de la distribución de frecuencias de la señal EMG. Su aplicación permite identificar fenómenos como la fatiga muscular, cambios en la conducción de las fibras y variaciones en la activación de las unidades motoras, proporcionando información cuantitativa y objetiva complementaria al análisis temporal. No obstante, su precisión depende de una adecuada adquisición y procesamiento de la señal, ya que el ruido y los artefactos pueden alterar los resultados. El análisis espectral ofrece un medio no invasivo, sensible  para el diagnóstico, la rehabilitación y el monitoreo del rendimiento muscular


