import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from scipy.stats import poisson

#---------------------------------------------
# PARTE 1: PROBLEMA DE CANCELACIÓN DE CUENTAS 
# EN UNA APLICACIÓN WEB
#---------------------------------------------

#se extrae la tabla del archivo .csv
datos = pd.read_csv('cancelaciones.csv')
#se extrae la columna cancelaciones
cancelaciones = datos['cancelaciones']

#---------------------------------------------
# 1.1. Crear una tabla que incluya frecuencias absolutas, 
# probabilidades emp´ıricas, y distribuciones acumuladas 
# empíricas, para la cantidad de cancelaciones diarias.
# value_counts cuenta la cantidad de ocurrencias para cada 
# cantidad de cancelaciones, generando una serie (parecido a un diccionario)
# sort_index ordena en forma ascendente la cantidad de cancelaciones 
# reset_index() convierte la anterior serie generada en un DataFrame. 
# DataFrame: tabla de datos, como un Excel, pero para Python. 
tabla = cancelaciones.value_counts().sort_index().reset_index()
tabla.columns = ['cancelaciones', 'frecuencia']
tabla['probabilidad'] = tabla['frecuencia'] / len(cancelaciones)
tabla['acumulada'] = tabla['probabilidad'].cumsum()

print("\n1.1. Tabla de frecuencias, probabilidades empíricas y distribución acumulada:")
print(tabla)

#---------------------------------------------
# 1.2. Calcular la esperanza y varianza empíricas 
# de la cantidad de cancelaciones diarias.
esperanza = cancelaciones.mean()
varianza = cancelaciones.var(ddof=0)

print("1.2. Esperanza y varianza empíricas:")
print("Esperanza empírica: ", esperanza)
print("Varianza empírica: ", varianza)

#---------------------------------------------
# 1.3.  Calcular la mediana y el rango intercuartílico 
# para la cantidad de cancelaciones diarias. Crear un 
# diagrama de cajas para la cantidad de cancelaciones diarias. 
mediana = cancelaciones.median()
q1 = cancelaciones.quantile(0.25)
q3 = cancelaciones.quantile(0.75)
rango_intercuartilico = q3 - q1

print("1.3. Mediana y rango intercuartílico para cantidad de cancelaciones diarias: ")
print("Mediana: ", mediana)
print("Rango intercuartílico: ", rango_intercuartilico)

# DIAGRAMA DE CAJAS 
# se define un tamaño en específico para el gráfico 
plt.figure(figsize=(6, 4))
# se genera un diagrama de cajas con mediana, cuartiles, 
# y valores atípicos
sns.boxplot(x=cancelaciones, color="pink")
# titulo para el gráfico 
plt.title("Diagrama de cajas")
# titulo para el eje de las x 
plt.xlabel("Cancelaciones")
plt.show()

#---------------------------------------------
# 1.4. Crear un histograma para la cantidad de 
# cancelaciones diarias.
plt.figure(figsize=(8, 5))
sns.histplot(
    cancelaciones, 
    # range se asegura de que haya una barra por 
    # cada número entero dentro de cancelaciones 
    bins=range(cancelaciones.min(), 
               cancelaciones.max()+2), 
    # no se dibuja la curva de densidad
    kde=False, 
    color="pink", 
    edgecolor="grey")
plt.title("Histograma de cancelaciones diarias")
plt.xlabel("Cancelaciones")
plt.ylabel("Frecuencia")
plt.show()

#---------------------------------------------
# 1.5. En el histograma de cancelaciones diarias, 
# superponga un histograma con la función de probabilidad 
# de masa de una variable aleatoria con distribución de 
# Poisson con un parámetro elegido cuidadosamente. 
# ¿Es razonable este modelo?

lambda_poisson = esperanza  #se usa la media empírica

# generación de los valores enteros de cancelaciones para 
# la función de Poisson
valores_x = np.arange(cancelaciones.min(), cancelaciones.max() + 1)
# cada probabilidad puntual se multilpica por la cantidad 
# de diías para escalar las probabilidades a frecuencias 
# abosultas y así poder compararlas con el histograma
probabilidades_poisson = poisson.pmf(valores_x, mu=lambda_poisson) * len(cancelaciones)

plt.figure(figsize=(8, 5))
sns.histplot(
    cancelaciones, 
    bins=range(cancelaciones.min(), 
               cancelaciones.max()+2), 
    color="pink", 
    edgecolor="grey", 
    stat="frequency", 
    label="Datos empíricos")
plt.plot(
    valores_x, 
    probabilidades_poisson, 
    'o-', #linea conectando los puntos
    color='purple', 
    label=f'Poisson(λ={lambda_poisson:.2f})')
plt.title("Histograma con ajuste Poisson")
plt.xlabel("Cancelaciones por día")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

#---------------------------------------------
# 1.6. Utilice el modelo anterior para calcular la 
# probabilidad de que en un cierto día la aplicación tenga
# menos de 5 cancelaciones, y la probabilidad de que en un 
# cierto día la aplicación tenga más de 15 cancelaciones.

# cdf: función de distribución acumulada
prob_menor_5 = poisson.cdf(4, mu=lambda_poisson)
prob_mas_15 = 1 - poisson.cdf(15, mu=lambda_poisson)

print(f"1.6. Siendo X~P(λ={lambda_poisson:.2f})")
print(f"P(X < 5) = {prob_menor_5}")
print(f"P(X > 15) = {prob_mas_15}")