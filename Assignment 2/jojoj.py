import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the function f(x) = 4 - x^2
def f_x(x):
    return 4 - x**2

# Define parameters
L = 2
a = 3  # Since u_tt = 9 * u_xx => a^2 = 9 => a = 3
n_terms = 10  # Number of terms in the Fourier series

# Define the Fourier cosine coefficients α_n
def alpha_n(n):
    result, _ = quad(lambda x: f_x(x) * np.cos((2*n-1) * np.pi * x / (2*L)), 0, L)
    return (2 / L) * result

# Calculate the Fourier series approximation for u(x,t)
def u_xt(x, t, n_terms=10):
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        alpha = alpha_n(n)
        u += alpha * np.cos((2*n-1) * np.pi * x / (2*L)) * np.cos((2*n-1) * np.pi * a * t / (2*L))
    return u

# Set up the x-values and time points for plotting
x_vals = np.linspace(0, L, 100)
time_points = [0, 0.01, 0.05, 0.1]  # Different time instances

# Plot the solution at different time points
plt.figure(figsize=(10, 6))
for t in time_points:
    u_vals = u_xt(x_vals, t, n_terms)
    plt.plot(x_vals, u_vals, label=f't={t}')

plt.title('Solution of the Wave Equation $u(x,t)$ with Fourier Series Approximation')
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.legend()
plt.grid(True)
plt.show()