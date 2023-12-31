import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def calculate_descriptive_statistics(data):
    return {
        'mean': np.mean(data),
        'standard_deviation': np.std(data, ddof=1),
        'size': len(data)
    }


def calculate_pooled_standard_deviation(std1, std2, size1, size2):
    return np.sqrt((std1**2 / (size1) + std2**2 / (size2)))


def calculate_t_statistic(data1, data2, d0):
    stats1 = calculate_descriptive_statistics(data1)
    stats2 = calculate_descriptive_statistics(data2)
    mean_diff = stats1['mean'] - stats2['mean']
    pooled_std = calculate_pooled_standard_deviation(stats1['standard_deviation'],
                                                     stats2['standard_deviation'],
                                                     stats1['size'],
                                                     stats2['size'])
    return (mean_diff - d0) / (pooled_std)


def calculate_t_value(alpha, degrees_of_freedom):
    return stats.t.ppf(1 - alpha, degrees_of_freedom)


def calculate_degrees_of_freedom(data1, data2):
    size1, size2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    return ((var1 / size1 + var2 / size2)**2) / \
           ((var1**2 / ((size1 - 1) * size1**2)) +
            (var2**2 / ((size2 - 1) * size2**2)))


def calculate_confidence_interval(mean_diff, pooled_std, degrees_of_freedom, alpha):
    t_value = calculate_t_value(alpha / 2, degrees_of_freedom)
    margin_of_error = t_value * pooled_std
    return mean_diff - margin_of_error, mean_diff + margin_of_error


def calculate_tail_probability(t_statistic, degrees_of_freedom, two_tailed):
    p_value = stats.t.sf(np.abs(t_statistic), degrees_of_freedom)
    return p_value * 2 if two_tailed else p_value


def plot_t_distribution(t_stat, df, alpha, two_tailed):
    x = np.linspace(-5, 5, 1000)
    y = stats.t.pdf(x, df)
    plt.plot(x, y, label=f'T-distribution with {df} DoF')

    if two_tailed:
        crit_val1 = stats.t.ppf(alpha / 2, df)
        crit_val2 = stats.t.ppf(1 - alpha / 2, df)
        plt.fill_between(x, y, where=(x < crit_val1) | (
            x > crit_val2), color='red', alpha=0.5)
    else:
        crit_val = stats.t.ppf(1 - alpha, df)
        plt.fill_between(x, y, where=x > crit_val, color='red', alpha=0.5)

    plt.axvline(t_stat, color='green', linestyle='--',
                label=f'T-statistic: {t_stat}')
    plt.legend()
    plt.xlabel('t-value')
    plt.ylabel('Probability Density')
    plt.title('T-Distribution and Critical Region')
    plt.grid(True)
    plt.show()


file_csv = 'AirDelay.csv'
df = pd.read_csv(file_csv)

# Supongamos que no conocemos los nombres de las columnas, los detectamos automáticamente
columns = df.columns
if len(columns) != 2:
    raise ValueError("El archivo CSV debe tener exactamente dos columnas.")

data1 = df[columns[0]].dropna().values
data2 = df[columns[1]].dropna().values

data1 = np.array(data1)
data2 = np.array(data2)

alpha = 0.05
two_tailed = False
d0 = 0

stats_data1 = calculate_descriptive_statistics(data1)
stats_data2 = calculate_descriptive_statistics(data2)

degrees_of_freedom = calculate_degrees_of_freedom(data1, data2)
t_statistic = calculate_t_statistic(data1, data2, d0)
tail_probability = calculate_tail_probability(
    t_statistic, degrees_of_freedom, two_tailed)

mean_diff = stats_data1['mean'] - stats_data2['mean']
pooled_std = calculate_pooled_standard_deviation(stats_data1['standard_deviation'],
                                                 stats_data2['standard_deviation'],
                                                 stats_data1['size'],
                                                 stats_data2['size'])
confidence_interval = calculate_confidence_interval(
    mean_diff, pooled_std, degrees_of_freedom, alpha)

print("Estadísticas de data1:")
for key, value in stats_data1.items():
    print(f"{key}: {value}")

print("\nEstadísticas de data2:")
for key, value in stats_data2.items():
    print(f"{key}: {value}")

print(f"\nDiferencia de medias: {mean_diff}")

print("\nIntervalo de confianza")
print(
    f"Con una confianza del {100 - alpha * 100}% se espera que la diferencia entre las medias esté entre {confidence_interval[0]} y {confidence_interval[1]}.")

print(f"\nEstadístico t: {t_statistic}")
print(f"Grados de libertad: {degrees_of_freedom}")
print(f"Probabilidad colateral (p-valor): {tail_probability}")

# Graficar la distribución t y la región crítica
plot_t_distribution(t_statistic, degrees_of_freedom, alpha, two_tailed)
