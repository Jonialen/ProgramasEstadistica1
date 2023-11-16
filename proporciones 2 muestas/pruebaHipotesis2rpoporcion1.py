import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def calcular_probabilidad_colateral_z(estadistico_z):
    probabilidad_colateral = scipy.stats.norm.sf(abs(estadistico_z))
    return probabilidad_colateral


def calcular_estadistico_z(proporcion1, proporcion2, tamaño1, tamaño2):
    prop_pooled = (proporcion1 * tamaño1 + proporcion2 *
                   tamaño2) / (tamaño1 + tamaño2)
    desviacion_pooled = np.sqrt(
        prop_pooled * (1 - prop_pooled) * (1/tamaño1 + 1/tamaño2))
    estadistico_z = (proporcion1 - proporcion2) / desviacion_pooled
    return estadistico_z


def plot_z_distribution_and_critical_region(z_stat, alpha):
    x = np.linspace(-5, 5, 1000)
    y = scipy.stats.norm.pdf(x)
    plt.plot(x, y, label='Distribución Z (Normal Estándar)')
    z_critical = scipy.stats.norm.ppf(1 - alpha)

    # Colorear la región central en rojo
    x_fill = np.linspace(z_critical, 5, 1000)
    y_fill = scipy.stats.norm.pdf(x_fill)
    plt.fill_between(x_fill, y_fill, color='red', alpha=0.5,
                     label=f'Región central ({alpha})')

    plt.axvline(z_stat, color='green', linestyle='--',
                label=f'Estadístico Z calculado: {z_stat}')
    plt.legend()
    plt.xlabel('Valor de Z')
    plt.ylabel('Densidad de probabilidad')
    plt.title('Distribución Z y Región Central (Normal Estándar)')
    plt.grid(True)
    plt.show()


# Datos de muestra
muestra1 = 12
muestra2 = 8
tamaño1 = 56
tamaño2 = 32
proporcion1 = muestra1/tamaño1
proporcion2 = muestra2/tamaño2
alpha = 0.01
h0 = 0

# Calcular estadístico Z
estadistico_z = calcular_estadistico_z(
    proporcion1, proporcion2, tamaño1, tamaño2)


print(f"Proporción agrupada: {proporcion1} y {proporcion2}")
print(f"Estadístico Z: {estadistico_z}")
print(
    f"Probabilidad que queda en la región central de Z: {calcular_probabilidad_colateral_z(estadistico_z)}")

# Graficar la distribución Z y la región central en rojo
plot_z_distribution_and_critical_region(estadistico_z, alpha)
