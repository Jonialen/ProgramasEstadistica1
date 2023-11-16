import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def calcular_intervalo_confianza_f(data1, data2, alpha):
    varianza1, tamaño1 = data1
    varianza2, tamaño2 = data2

    f_critico_inferior = scipy.stats.f.ppf(alpha / 2, tamaño1 - 1, tamaño2 - 1)
    f_critico_superior = scipy.stats.f.ppf(
        1 - alpha / 2, tamaño1 - 1, tamaño2 - 1)

    ratio_inferior = varianza1 / varianza2 * f_critico_inferior
    ratio_superior = varianza1 / varianza2 * f_critico_superior

    return (ratio_inferior, ratio_superior)


def calcular_estadistico_f(data1, data2):
    varianza1, tamaño1 = data1
    varianza2, tamaño2 = data2

    return varianza1 / varianza2


def calcular_probabilidad_colateral_f(estadistico_f, tamaño1, tamaño2):
    return scipy.stats.f.sf(estadistico_f, tamaño1 - 1, tamaño2 - 1)


def graficar_distribucion_f_y_region_critica(estadistico_f, tamaño1, tamaño2, alpha):
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


# Ejemplo de uso del código para la prueba F
varianza1 = 1.812929293
tamaño1 = 100

varianza2 = 0.998645455
tamaño2 = 100

data1 = (varianza1, tamaño1)
data2 = (varianza2, tamaño2)

if varianza1 < varianza2:
    data1, data2 = data2, data1

alpha = 0.01
estadistico_f = calcular_estadistico_f(data1, data2)
probabilidad_colateral_f = calcular_probabilidad_colateral_f(
    estadistico_f, tamaño1, tamaño2)

intervalo_confianza_f = calcular_intervalo_confianza_f(data1, data2, alpha)

# Mostrar resultados
print(f"Intervalo de confianza F: {intervalo_confianza_f}")
print(f"Estadístico F: {estadistico_f}")
print(f"Probabilidad colateral F: {probabilidad_colateral_f}")

graficar_distribucion_f_y_region_critica(
    estadistico_f, tamaño1, tamaño2, alpha)
