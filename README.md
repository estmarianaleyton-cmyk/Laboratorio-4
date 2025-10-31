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

## **Resultados de frecuencias por contración:**
<img width="519" height="121" alt="image" src="https://github.com/user-attachments/assets/7163de94-881d-41a0-b8f8-8f43a9802630" />

## **Gráfica de la evolución de las frecuencias por contracción**
<img width="887" height="550" alt="image" src="https://github.com/user-attachments/assets/60f6c1ad-4de4-4d3d-9700-6e877bcfe5e7" />

## **Análisis de la variación de las frecuencias a lo largo de las contracciones simuladas**
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
voltaje = np.loadtxt("EMG_señal.txt")

# Definir parámetros de muestreo
fs = 1500
N = len(voltaje)
t = np.arange(N) / fs  # eje de tiempo

# Graficar
plt.figure(figsize=(10,4))
plt.plot(t, voltaje, label="Señal EOG, tomada en el laboratorio")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title("Señal EMG tomada en el laboratorio ")
plt.grid(True)
plt.show()
  ´´´
</pre>
## **Diagrama de flujo**

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

## **Gráfica de la señal EMG tomada en el laboratorio con filtro pasa banda**
<img width="1219" height="306" alt="image" src="https://github.com/user-attachments/assets/00db37fc-35c6-4591-9cba-7c5cf6a5da18" />

## **Código en Python (Google colab)**
<pre> ´´´
# Frecuencia media y mediana
# --- 1. Duración de cada ventana de contracción ---
ventana = 0.08  # segundos (80 ms)
muestras_ventana = int(ventana * fs / 2)  # mitad antes y mitad después

frecuencia_media = []
frecuencia_mediana = []
indices_contracciones = []

# --- 2. Calcular frecuencia media y mediana para cada contracción detectada ---
for p in peaks:
    ini = max(0, p - muestras_ventana)
    fin = min(len(voltaje_filt), p + muestras_ventana)
    segmento = voltaje_filt[ini:fin]

    # Evitar análisis de ventanas vacías
    if len(segmento) < 10:
        continue

    # --- 3. Espectro de potencia mediante Welch ---
    f, Pxx = welch(segmento, fs=fs, nperseg=min(256, len(segmento)))

    # --- 4. Calcular frecuencia media y mediana ---
    f_media = np.sum(f * Pxx) / np.sum(Pxx)
    f_acum = np.cumsum(Pxx)
    f_mediana = f[np.where(f_acum >= np.sum(Pxx)/2)[0][0]]

    frecuencia_media.append(f_media)
    frecuencia_mediana.append(f_mediana)
    indices_contracciones.append(p / fs)  # tiempo en segundos de la contracción

# --- 5. Mostrar resultados numéricos ---
for i, (fm, fmed) in enumerate(zip(frecuencia_media, frecuencia_mediana)):
    print(f"Contracción {i+1}: Frecuencia media = {fm:.2f} Hz, Frecuencia mediana = {fmed:.2f} Hz")

# --- 6. Graficar evolución de las frecuencias ---
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

## **Gráfica de la señal EMG con frecuencia media y mediana**
<img width="881" height="456" alt="image" src="https://github.com/user-attachments/assets/18fae2ff-c06c-4053-967d-6b36c8b99240" />
## **Resultados**





