import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def calcular_intervalo_de_confiza(data1, data2, grados_libertad, alpha, isDosColas):
    alpha = alpha*2 if isDosColas else alpha
    estadisticas_data1 = calcular_estadistica_descriptiva(data1)
    estadisticas_data2 = calcular_estadistica_descriptiva(data2)

    desviacion1 = estadisticas_data1['desviacion_estandar']
    desviacion2 = estadisticas_data2['desviacion_estandar']

    tamaño1 = estadisticas_data1['tamaño']
    tamaño2 = estadisticas_data2['tamaño']

    media1 = estadisticas_data1['media']
    media2 = estadisticas_data2['media']

    desviacion_pooled = calcular_desviacion_pooled(
        desviacion1, desviacion2, tamaño1, tamaño2)

    dif = media1-media2 if media1 > media2 else media2-media1
    valor_t = calcular_valor_t(alpha/2, grados_libertad)
    menor = dif-valor_t*desviacion_pooled
    mayor = dif+valor_t*desviacion_pooled

    return (menor, mayor)


def calcular_valor_t(alpha, grados_libertad):
    valor_t = scipy.stats.t.ppf(1 - alpha, grados_libertad)
    return valor_t


def calcular_probabilidad_colateral_t(estadistico_t, grados_libertad, isDosColas):
    probabilidad_colateral = scipy.stats.t.sf(estadistico_t, grados_libertad)
    probabilidad_colateral = probabilidad_colateral * \
        2 if isDosColas else probabilidad_colateral
    return probabilidad_colateral


def calcular_estadistica_descriptiva(data):
    estadisticas = {
        'media': data[0],
        'desviacion_estandar': data[1],
        'tamaño': data[2]
    }
    return estadisticas


def calcular_desviacion_pooled(desviacion1, desviacion2, tamaño1, tamaño2):
    desviacion_pooled = ((desviacion1**2/tamaño1) +
                         (desviacion2**2/tamaño2))**0.5
    return desviacion_pooled


def calcular_estadistico_t(data1, data2, d0):
    estadisticas_data1 = calcular_estadistica_descriptiva(data1)
    estadisticas_data2 = calcular_estadistica_descriptiva(data2)

    desviacion1 = estadisticas_data1['desviacion_estandar']
    desviacion2 = estadisticas_data2['desviacion_estandar']

    tamaño1 = estadisticas_data1['tamaño']
    tamaño2 = estadisticas_data2['tamaño']

    media1 = estadisticas_data1['media']
    media2 = estadisticas_data2['media']

    desviacion_pooled = calcular_desviacion_pooled(
        desviacion1, desviacion2, tamaño1, tamaño2)

    dif = media1-media2
    estadistico_t = (dif-d0) / desviacion_pooled
    return estadistico_t


def calcular_grados_de_libertad(data1, data2):
    tamaño1 = data1[2]
    tamaño2 = data2[2]
    varianza1 = data1[1]**2
    varianza2 = data2[1]**2
    grados_libertad = ((varianza1 / tamaño1 + varianza2 / tamaño2)**2) / ((varianza1 **
                                                                           2 / ((tamaño1 - 1) * tamaño1**2)) + (varianza2 / ((tamaño2 - 1) * tamaño2**2)))
    return grados_libertad


def plot_t_distribution_and_critical_region(t_stat, df, alpha, isDosColas):

    x = np.linspace(-5, 5, 1000)
    y = scipy.stats.t.pdf(x, df)
    plt.plot(x, y, label=f'Distribución t con {df} grados de libertad')
    if isDosColas:
        t_critical_1 = scipy.stats.t.ppf(alpha, df)
        t_critical_2 = scipy.stats.t.ppf(1 - alpha, df)
        x_fill = np.linspace(-5, t_critical_1, 1000)
        y_fill = scipy.stats.t.pdf(x_fill, df)
        plt.fill_between(x_fill, y_fill, color='red', alpha=0.5,
                         label=f'Región crítica ({alpha})')

        x_fill = np.linspace(t_critical_2, 5, 1000)
        y_fill = scipy.stats.t.pdf(x_fill, df)
        plt.fill_between(x_fill, y_fill, color='red', alpha=0.5,
                         label=f'Región crítica ({alpha})')

    else:
        t_critical = scipy.stats.t.ppf(1 - alpha/2, df)
        x_fill = np.linspace(t_critical, 5, 1000)
        y_fill = scipy.stats.t.pdf(x_fill, df)
        plt.fill_between(x_fill, y_fill, color='red', alpha=0.5,
                         label=f'Región crítica ({alpha})')

    plt.axvline(t_stat, color='green', linestyle='--',
                label=f'Estadístico t calculado: {t_stat}')
    plt.legend()
    plt.xlabel('Valor de t')
    plt.ylabel('Densidad de probabilidad')
    plt.title('Distribución t y Región Crítica')
    plt.grid(True)
    plt.show()


# Datos de muestra
media1 = 22.1
media2 = 20.4
desviacion1 = 4.09
desviacion2 = 3.08
tamaño1 = 11
tamaño2 = 7

data1 = [media1, desviacion1, tamaño1]
data2 = [media2, desviacion2, tamaño2]

data1 = np.array(data1)
data2 = np.array(data2)

estadisticas_data1 = calcular_estadistica_descriptiva(data1)
estadisticas_data2 = calcular_estadistica_descriptiva(data2)

d0 = 0
alpha = 0.05
isDosColas = False
alpha = alpha/2 if isDosColas else alpha

# Calcular estadístico t y grados de libertad
estadistico_t = calcular_estadistico_t(data1, data2, d0)
grados_libertad = calcular_grados_de_libertad(data1, data2)

# Calcular la probabilidad que deja la cola del valor de t
probabilidad_colateral_t = calcular_probabilidad_colateral_t(
    abs(estadistico_t), grados_libertad, isDosColas)

intervalo_de_confiza = calcular_intervalo_de_confiza(
    data1, data2, grados_libertad, alpha, isDosColas)
# Graficar la distribución t y la región crítica
print("Estadísticas de data1:")
for key, value in calcular_estadistica_descriptiva(data1).items():
    print(f"{key}: {float(value)}")

print("\nEstadísticas de data2:")
for key, value in calcular_estadistica_descriptiva(data2).items():
    print(f"{key}: {float(value)}")

print(
    f"\nDiferencia de medias: {estadisticas_data1['media']-estadisticas_data2['media']}")
print("\nIntervalo de confiza")
print(
    f"Con una confianza de {100-alpha*200 if isDosColas else 100-alpha*100}%, la diferencia entre las medias se encuentra entre {intervalo_de_confiza[0].round(4)} y {intervalo_de_confiza[1].round(4)}.")

print(f"\nEstadístico t: {estadistico_t.round(4)}")
print(f"Grados de libertad: {grados_libertad.round(4)}")
print(f"Valor-P: {probabilidad_colateral_t.round(4)}")

plot_t_distribution_and_critical_region(
    estadistico_t, grados_libertad, alpha, isDosColas)
