import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## Problem 2: Maximum likelihood, minsta kvadrat
M = 10000
b = 4
# Simulera M utfall med parameter b.
x = stats.rayleigh.rvs(scale=b, size=M)
# Skapa figur och plotta histogrammet.
plt.figure()
plt.hist(x, 40, density=True)
est_ml = np.sqrt((1 / (2 * M)) * np.sum(x**2))  # Skriv din ML-skattning här
est_mk = (np.mean(x)) / (np.sqrt(np.pi / 2))  # Skriv din MK-skattning här
# Plotta de två skattningarna.
plt.plot(est_mk, 0.1, "g*", markersize=10)
plt.plot(est_ml, 0.2, "r*", markersize=10)
plt.plot(b, 0.09, "bo")
plt.show()

## Problem 2: Maximum likelihood, minsta kvadrat (forts.)
# Skapa figur.
plt.figure()
# Visa histogrammet.
plt.hist(x, 40, density=True)
# Plotta täthetsfunktionen för den skattade parametern.
x_grid = np.linspace(np.min(x), np.max(x), 60)
pdf = stats.rayleigh.pdf(x_grid, scale=est_ml)
plt.plot(x_grid, pdf, "r")
plt.show()
