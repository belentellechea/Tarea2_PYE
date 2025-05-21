import pandas as pd
import numpy as np

#Parte 1A

file = pd.read_csv("cancelaciones.csv")
data = file['cancelaciones']

frecuencia_absoluta = data.value_counts().sort_index()

probabilidad_empirica = frecuencia_absoluta / frecuencia_absoluta.sum()

distribucion_acumulada = probabilidad_empirica.cumsum()

tabla = pd.DataFrame({
    'Frecuencia absoluta': frecuencia_absoluta,
    'Probabilidad empírica': probabilidad_empirica,
    'Distribución acumulada': distribucion_acumulada
})
print(tabla)

#Parte 1B
