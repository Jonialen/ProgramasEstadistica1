import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def calcular_estadisticas_muestra(data):
    varianza = np.var(data, ddof=1)  # Varianza de la muestra
    tamaño = len(data)  # Tamaño de la muestra
    return varianza, tamaño


def calcular_intervalo_confianza_f(data1, data2, alpha):
    varianza1, tamaño1 = calcular_estadisticas_muestra(data1)
    varianza2, tamaño2 = calcular_estadisticas_muestra(data2)

    f_critico_inferior = scipy.stats.f.ppf(alpha / 2, tamaño1 - 1, tamaño2 - 1)
    f_critico_superior = scipy.stats.f.ppf(
        1 - alpha / 2, tamaño1 - 1, tamaño2 - 1)

    ratio_inferior = varianza1 / varianza2 * f_critico_inferior
    ratio_superior = varianza1 / varianza2 * f_critico_superior

    return (ratio_inferior, ratio_superior)


def calcular_estadistico_f(data1, data2):
    varianza1, tamaño1 = calcular_estadisticas_muestra(data1)
    varianza2, tamaño2 = calcular_estadisticas_muestra(data2)

    return varianza1 / varianza2


def calcular_probabilidad_colateral_f(estadistico_f, data1, data2):
    _, tamaño1 = calcular_estadisticas_muestra(data1)
    _, tamaño2 = calcular_estadisticas_muestra(data2)

    return scipy.stats.f.sf(estadistico_f, tamaño1 - 1, tamaño2 - 1)


def graficar_distribucion_f_y_region_critica(estadistico_f, data1, data2, alpha):
    _, tamaño1 = calcular_estadisticas_muestra(data1)
    _, tamaño2 = calcular_estadisticas_muestra(data2)
    dfn, dfd = tamaño1 - 1, tamaño2 - 1

    x = np.linspace(0, 5, 1000)
    y = scipy.stats.f.pdf(x, dfn, dfd)
    plt.plot(
        x, y, label=f'Distribución F con {dfn} y {dfd} grados de libertad')

    f_critico = scipy.stats.f.ppf(1 - alpha, dfn, dfd)
    x_critico = np.linspace(f_critico, 5, 1000)
    y_critico = scipy.stats.f.pdf(x_critico, dfn, dfd)
    plt.fill_between(x_critico, y_critico, color='red', alpha=0.5,
                     label=f'Región crítica ({alpha})')

    plt.axvline(estadistico_f, color='green', linestyle='--',
                label=f'Estadístico F: {estadistico_f}')
    plt.legend()
    plt.xlabel('Valor F')
    plt.ylabel('Densidad de probabilidad')
    plt.title('Distribución F y Región Crítica')
    plt.grid(True)
    plt.show()


# Ejemplo de uso del código con listas de datos
lista1 = []
lista2 = []
datos1 = np.array(lista1)
datos2 = np.array(lista2)

# Asegurarse de que datos1 tenga la varianza más alta
varianza1, _ = calcular_estadisticas_muestra(datos1)
varianza2, _ = calcular_estadisticas_muestra(datos2)

if varianza2 > varianza1:
    datos1, datos2 = datos2, datos1

alpha = 0.05
estadistico_f = calcular_estadistico_f(datos1, datos2)
probabilidad_colateral_f = calcular_probabilidad_colateral_f(
    estadistico_f, datos1, datos2)

intervalo_confianza_f = calcular_intervalo_confianza_f(datos1, datos2, alpha)

# Mostrar resultados
print(f"Intervalo de confianza F: {intervalo_confianza_f}")
print(f"Estadístico F: {estadistico_f}")
print(f"Probabilidad colateral F: {probabilidad_colateral_f}")

graficar_distribucion_f_y_region_critica(
    estadistico_f, datos1, datos2, alpha)
