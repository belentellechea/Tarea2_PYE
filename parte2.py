import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import cauchy

#---------------------------------------------
# PARTE 2: SIMULACIÓN DE VARIABLES ALEATORIAS
# CONTINUAS
#---------------------------------------------

# 2.1. Algoritmo para simular una variable aleatoria U ~ U[0,1]

# La formula brindada en clase es un generador congruencial lineal (LGC)
# este genera una secuencia de nùmeros enteros que parecen aleatorios, pero que en realidad siguen una fòrmula matemàtica. 
# Luego se normalizan dividiendo entre un nùmero grande para obtener nùmeros en el intervalo [0,1], como si fueran variables aleatorias U[0,1]

def generador_uniforme(seed, a=1664525, c=1013904223, m=2**32):
    # a es el multiplicador, c es el incremento, m es el módulo
    # el modulo define el rango màximo de los valores (perìodo)
    x = (a * seed + c) % m
    u = x / m
    return x, u

# los valores seleccionados son clasicos en implementaciones de LCG (se usan en la biblioteca estándar de C)
# los mismos fueron recomendados por Donald Knuth en su libro "The Art of Computer Programming"
#Son valores ampliamente utilizados dado su buen comportamiento estadistico y el largo periodo (de 2^32)

# Usamos reloj para obtener semilla inicial
seed = int(time.time())
muestra_uniforme = []

# Generar 100 números U[0,1]
for _ in range(100):
    seed, u = generador_uniforme(seed)
    #la semilla se actualiza en cada iteración
    # Guardamos el número x generado
    muestra_uniforme.append(u)

# Convertimos a arreglo de NumPy
muestra_uniforme = np.array(muestra_uniforme)

# Histograma + curva de densidad estimada
plt.figure(figsize=(8, 5))
#sns.histplot(muestra_uniforme, bins=10, kde=True, color='pink', edgecolor='grey')
sns.histplot(muestra_uniforme, bins=10, color='pink', edgecolor='grey', stat='density')

# Graficar la curva KDE por separado, en hot pink
sns.kdeplot(muestra_uniforme, color='#ff006e')

plt.title("Histograma y estimación de densidad - U[0,1]")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()


# 2.4 Función para simular una variable Cauchy estándar a partir de U[0,1]
def inversa_cauchy(u):
    return np.tan(np.pi * (u - 0.5))

# Generar muestra de tamaño 100 con la inversa aplicada a la muestra uniforme generada
muestra_cauchy = np.array([inversa_cauchy(u) for u in muestra_uniforme])


# 2.5 Histograma + densidad estimada + densidad teórica de Cauchy
plt.figure(figsize=(10, 6))

# Histograma y KDE
#sns.histplot(muestra_cauchy, bins=30, kde=True, stat="density", color='pink', edgecolor='grey', label='Estimación KDE')
sns.histplot(muestra_cauchy, bins=30, stat="density", color='pink', edgecolor='grey', label='Estimación KDE')
# Graficar la curva KDE por separado, en hot pink
sns.kdeplot(muestra_cauchy, color='#ff006e')

# Densidad teórica de Cauchy
x_vals = np.linspace(-10, 10, 1000)
densidad_cauchy = cauchy.pdf(x_vals)
plt.plot(x_vals, densidad_cauchy, color='#a4133c', label='Densidad Cauchy teórica')

plt.title("Distribución Cauchy estándar: Histograma + densidad estimada y teórica")
plt.xlabel("Valor")
plt.ylabel("Densidad")
plt.legend()
plt.grid(True)
plt.show()
