import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm

# -------------------------------
# Spatial and Time Discretization
# -------------------------------
dx = 0.1
dt = 0.01
L = 100
x = np.linspace(-L/2, L/2, int(L/dx))
n_x = len(x)

# -------------------------------
# Trefoil Knot-Inspired Wavefunction
# -------------------------------
def trefoil_wavefunction(x):
    return np.exp(-x**2) * (np.exp(1j * 3 * x) - np.exp(1j * 4 * x) + np.exp(1j * x))

# -------------------------------
# Potential: Rectangular Barrier
# -------------------------------
def rectangular_barrier_potential(x, V0=10, a=5):
    V = np.zeros_like(x)
    V[np.abs(x) < a] = V0
    return V

V = rectangular_barrier_potential(x)
H_potential = np.diag(V)

# -------------------------------
# Kinetic Operator with Periodic Boundary Conditions
# -------------------------------
H_free = - np.diag(np.ones(n_x - 1), -1) - np.diag(np.ones(n_x - 1), 1)
H_free[0, n_x - 1] = -1
H_free[n_x - 1, 0] = -1
H_free *= -1 / (2 * dx**2)

# -------------------------------
# Reidemeister Move Operators
# -------------------------------
def R1(psi):
    return np.roll(psi, 1)

def R2(psi):
    return np.roll(psi, -1)

def R3(psi):
    return np.roll(psi, 2)

# Reidemeister (topological) perturbation
H_reidemeister = np.zeros((n_x, n_x), dtype=complex)
for i in range(n_x - 1):
    H_reidemeister[i, (i + 1) % n_x] += 0.05
    H_reidemeister[(i + 1) % n_x, i] += 0.05

# -------------------------------
# Total Hamiltonian & Unitary Evolution
# -------------------------------
H = H_free + H_potential + H_reidemeister
U = expm(-1j * H * dt)

# Initialize wavefunction
psi = trefoil_wavefunction(x)
psi /= np.linalg.norm(psi)

# Apply Reidemeister sequence
U_reidemeister = np.eye(n_x, dtype=complex)
def apply_reidemeister_sequence(psi, sequence):
    global U_reidemeister
    for move in sequence:
        psi = move(psi)
    U_reidemeister = U_reidemeister @ expm(-1j * H_reidemeister * dt)
    return psi

# -------------------------------
# Plotting in a Ring (Polar Plot)
# -------------------------------
theta = np.linspace(0, 2 * np.pi, n_x, endpoint=False)
R0 = 1.0
scale_psi = 0.5
scale_V = 0.5
V_normalized = (V - np.min(V)) / (np.max(V) - np.min(V) + 1e-9)
r_V = R0 + scale_V * V_normalized

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.set_ylim(0, R0 + scale_psi + 0.1)
ax.set_title("Trefoil Knot Wavefunction with Rectangular Barrier")
line_psi, = ax.plot([], [], 'b', label=r'Wavefunction $|\psi|^2$')
line_V, = ax.plot(theta, r_V, 'r', label='Potential')
ax.legend(loc='upper right')

# -------------------------------
# Time Evolution (Animation)
# -------------------------------
n_steps = 2000
frames = 2000

def update(frame):
    global psi
    for _ in range(n_steps // frames):
        psi = U @ psi
        if np.random.uniform(0, 1) < 0.01:
            psi = apply_reidemeister_sequence(psi, [np.random.choice([R1, R2, R3])])
        psi /= np.linalg.norm(psi)
    psi_abs2 = np.abs(psi)**2
    psi_norm = psi_abs2 / np.max(psi_abs2)
    r_psi = R0 + scale_psi * psi_norm
    line_psi.set_data(theta, r_psi)
    return line_psi,

ani = animation.FuncAnimation(fig, update, frames=frames, interval=10, blit=True)
plt.show()