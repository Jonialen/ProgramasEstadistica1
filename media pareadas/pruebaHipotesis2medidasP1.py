import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def calcular_intervalo_de_confiza(data, grados_libertad, alpha, isDosColas):
    alpha = alpha*2 if isDosColas else alpha
    estadisticas_data = calcular_estadistica_descriptiva(data)

    desviacion = estadisticas_data['desviacion_estandar']

    tamaño = estadisticas_data['tamaño']

    media = estadisticas_data['media']

    desviacion_pooled = calcular_desviacion_pooled(desviacion, tamaño)

    valor_t = calcular_valor_t(alpha/2, grados_libertad)
    menor = media-valor_t*desviacion_pooled
    mayor = media+valor_t*desviacion_pooled

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


def calcular_desviacion_pooled(desviacion, tamaño):
    desviacion_pooled = desviacion/np.sqrt(tamaño)
    return desviacion_pooled


def calcular_estadistico_t(data, d0):
    estadisticas_data = calcular_estadistica_descriptiva(data)

    desviacion = estadisticas_data['desviacion_estandar']

    tamaño = estadisticas_data['tamaño']

    media = estadisticas_data['media']

    desviacion_pooled = calcular_desviacion_pooled(
        desviacion, tamaño)

    estadistico_t = (media-d0) / desviacion_pooled
    return estadistico_t


def calcular_grados_de_libertad(data):
    grados_libertad = data[2]-1
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


media = 22.1
desviacion = 40.9
tamaño = 11

data = np.array([media, desviacion, tamaño])

estadisticas_data = calcular_estadistica_descriptiva(data)

d0 = 0
alpha = 0.05
isDosColas = True
alpha = alpha/2 if isDosColas else alpha

# Calcular estadístico t y grados de libertad
estadistico_t = calcular_estadistico_t(data, d0)
grados_libertad = calcular_grados_de_libertad(data)

# Calcular la probabilidad que deja la cola del valor de t
probabilidad_colateral_t = calcular_probabilidad_colateral_t(
    abs(estadistico_t), grados_libertad, isDosColas)

intervalo_de_confiza = calcular_intervalo_de_confiza(
    data, grados_libertad, alpha, isDosColas)
# Graficar la distribución t y la región crítica
print("Estadísticas de data:")
for key, value in calcular_estadistica_descriptiva(data).items():
    print(f"{key}: {float(value)}")


print(
    f"\nDiferencia de medias: {estadisticas_data['media']-estadisticas_data['media']}")
print("\nIntervalo de confiza")
print(
    f"Con una confianza de {100-alpha*200 if isDosColas else 100-alpha*100}%, la diferencia entre las medias se encuentra entre {intervalo_de_confiza[0].round(4)} y {intervalo_de_confiza[1].round(4)}.")

print(f"\nEstadístico t: {estadistico_t.round(4)}")
print(f"Grados de libertad: {grados_libertad}")
print(f"Valor-P: {probabilidad_colateral_t}")

plot_t_distribution_and_critical_region(
    estadistico_t, grados_libertad, alpha, isDosColas)
