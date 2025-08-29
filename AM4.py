import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm
from scipy.integrate import quad  # for WKB if needed

# -------------------------------
# Spatial and Time Discretization
# -------------------------------
dx = 0.1
dt = 0.01
L = 100
x = np.linspace(-L/2, L/2, int(L/dx))
n_x = len(x)

# -------------------------------
# Define a Nontrivial Metric to Include Curvature
# -------------------------------
# The metric is given by g(x)=1+curvature_factor*cos(2*pi*x/L)
curvature_factor = 0.5  # adjust to increase/decrease curvature effects
g = 1.0 + curvature_factor * np.cos(2 * np.pi * x / L)

# For the Laplace–Beltrami operator, define:
# p(x) = 1/sqrt(g(x))
p = 1/np.sqrt(g)
# Compute p at half-integer points (using average of neighboring squared values)
p_half_sq = (p**2 + np.roll(p, -1)**2) / 2.

# -------------------------------
# Build the Kinetic Operator with Curvature (Laplace–Beltrami Discretization)
# -------------------------------
# The discretized operator (with periodic BCs) is:
# (H_free)_i = -1/(2 dx^2)[ p_{i+1/2}^2 ψ_{i+1} - (p_{i+1/2}^2 + p_{i-1/2}^2) ψ_i + p_{i-1/2}^2 ψ_{i-1} ]
H_free = np.zeros((n_x, n_x), dtype=complex)
for i in range(n_x):
    ip = (i + 1) % n_x
    im = (i - 1) % n_x
    H_free[i, ip] = -p_half_sq[i] / (2 * dx**2)
    H_free[i, im] = -p_half_sq[im] / (2 * dx**2)
    H_free[i, i] = (p_half_sq[i] + p_half_sq[im]) / (2 * dx**2)

# -------------------------------
# Reidemeister Move Operators (Unitary Transformations)
# -------------------------------
def R1(psi):  # Type I: Loop Addition/Removal
    return np.roll(psi, 1)

def R2(psi):  # Type II: Strand Sliding
    return np.roll(psi, -1)

def R3(psi):  # Type III: Braid Commutation
    return np.roll(psi, 2)

# -------------------------------
# Initial Wavefunction
# -------------------------------
def psi_initial(x):
    return np.exp(-x**2) * np.exp(1j * 5 * x)

# -------------------------------
# Potential Functions
# -------------------------------
# (A) Rectangular Barrier Potential
def rectangular_barrier_potential(x, V0=10, a=5):
    V = np.zeros_like(x)
    V[np.abs(x) < a] = V0
    return V

# (B) Delta Function Potential (approximation)
def delta_potential(x, V0=1000, width=dx):
    V = np.zeros_like(x)
    idx_center = np.argmin(np.abs(x))
    V[idx_center] = V0 / width  # approximates a delta spike
    return V

# (C) Finite Square Well Potential
def finite_square_well_potential(x, V0=10, a=5):
    V = np.zeros_like(x)
    V[np.abs(x) < a] = -V0
    return V

# (D) Eckart Potential
def eckart_potential(x, V0=10, a=0.2):
    return V0 / (np.cosh(a * x)**2)

# (E) Double Well Potential at Finite Temperature
def double_well_finite_temp_potential(x, a=0.01, b=1, temperature=300, noise_strength=0.1):
    base = a * (x**2 - b**2)**2
    np.random.seed(42)
    noise = noise_strength * np.random.normal(0, np.sqrt(temperature), len(x))
    return base + noise

# (F) Lennard-Jones Potential (12-6 potential)
def lennard_jones_potential(x, epsilon=1.0, sigma=1.0):
    r = np.abs(x) + 1e-9  # Avoid division by zero
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# (G) 12-6 Potential (identical to Lennard-Jones)
def potential_12_6(x, epsilon=1.0, sigma=1.0):
    r = np.abs(x) + 1e-9
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# (H) Godbeer et al. Potential (Morse-like, shifted)
def godbeer_potential(x, D_e=10, a=0.5, r_e=5):
    return D_e * (1 - np.exp(-a * (np.abs(x) - r_e)))**2 - D_e

# (I) Sitnitsky Potential (candidate double-well form)
def sitnitsky_potential(x, V0=10, a=0.5, b=1):
    return V0 * (np.tanh(a*(x + b)) - np.tanh(a*(x - b))) / 2

# Additional potentials from your previous list
def harmonic_potential(x, k=0.05):
    return 0.5 * k * x**2

def double_well_potential(x, a=10e-3, b=50):
    return a * (x**2 - b**2)**2

def random_potential(x, strength=5):
    np.random.seed(42)
    return np.random.uniform(-strength, strength, len(x))

def step_potential(x, height=10, width=5):
    V = np.zeros_like(x)
    V[np.abs(x) < width] = height
    return V

def quadratic_barrier_potential(x, height=10, width=5):
    return height * np.exp(- (x / width) ** 2)

def sine_potential(x, amplitude=5, frequency=0.2):
    return amplitude * np.sin(frequency * x)

# -------------------------------
# Select Potential Function
# -------------------------------
# Choose potential_type from:
# "rectangular_barrier", "delta", "finite_square_well", "eckart",
# "double_well_finite_temp", "lennard_jones", "12_6", "godbeer", "sitnitsky",
# "harmonic", "double_well", "random", "step", "quadratic_barrier", "sine"
potential_type = "rectangular_barrier_potential"  # Change as desired

#ptypes = ["rectangular_barrier_potential", 'quadratic_barrier', 'sine_potential', "double_well_potential", "eckart","lennard_jones", "12_6", "godbeer", "sitnitsky"]
ptypes = ["godbeer", "sitnitsky"]
for potential_type in ptypes:
    if potential_type == "rectangular_barrier_potential":
        V = rectangular_barrier_potential(x)
    elif potential_type == "delta":
        V = delta_potential(x)
    elif potential_type == "finite_square_well":
        V = finite_square_well_potential(x)
    elif potential_type == "eckart":
        V = eckart_potential(x)
    elif potential_type == "double_well_finite_temp":
        V = double_well_finite_temp_potential(x)
    elif potential_type == "lennard_jones":
        V = lennard_jones_potential(x)
    elif potential_type == "12_6":
        V = potential_12_6(x)
    elif potential_type == "godbeer":
        V = godbeer_potential(x)
    elif potential_type == "sitnitsky":
        V = sitnitsky_potential(x)
    elif potential_type == "harmonic":
        V = harmonic_potential(x)
    elif potential_type == "double_well_potential":
        V = double_well_potential(x)
    elif potential_type == "random":
        V = random_potential(x)
    elif potential_type == "step":
        V = step_potential(x)
    elif potential_type == "quadratic_barrier":
        V = quadratic_barrier_potential(x)
    elif potential_type == "sine_potential":
        V = sine_potential(x)
    else:
        raise ValueError("Unknown potential type")

    # -------------------------------
    # Hamiltonian & Unitary Evolution
    # -------------------------------

    # Define Hamiltonian components

    H_potential = np.diag(V)  # Potential term

    # Define Reidemeister transformation as a perturbation to the Hamiltonian
    H_reidemeister = np.zeros((n_x, n_x), dtype=complex)
    for i in range(n_x - 1):
        H_reidemeister[i, (i + 1) % n_x] #+= 0.05  # Small transition amplitude
        H_reidemeister[(i + 1) % n_x, i] #+= 0.05

    # Total Hamiltonian includes kinetic, potential, and topological terms
    H = H_free + H_potential + H_reidemeister

    # Compute unitary evolution operator
    U = expm(-1j * H * dt)  # Unitary time evolution

    # Initialize wavefunction
    psi = psi_initial(x)

    # Define Reidemeister sequence transformation
    U_reidemeister = np.eye(n_x, dtype=complex)
    def apply_reidemeister_sequence(psi, sequence):
        global U_reidemeister
        for move in sequence:
            psi = move(psi)
        U_reidemeister = U_reidemeister @ expm(-1j * H_reidemeister * dt)
        return psi

    # Compute tunneling probability as transition amplitude
    psi_K_prime = apply_reidemeister_sequence(psi.copy(), [R1, R2, R3])
    P_tunnel = lambda psi: np.abs(np.vdot(psi_K_prime, U_reidemeister @ psi))**2

    # -------------------------------
    # Set Up Ring Coordinates for Plotting
    # -------------------------------
    # Map the spatial index to an angle theta (0 to 2pi)
    theta = np.linspace(0, 2 * np.pi, n_x, endpoint=False)

    # Base radius for the ring and scaling factors for visualization
    R0 = 1.0
    scale_psi = 0.5  # amplitude scaling for wavefunction probability density
    scale_V = 0.5    # amplitude scaling for potential visualization

    # Precompute the static potential radial profile (normalized)
    V_normalized = (V - np.min(V)) / (np.max(V) - np.min(V) + 1e-9)
    r_V = R0 + scale_V * V_normalized

    # -------------------------------
    # Plotting in a Ring (Polar Plot)
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_ylim(0, R0 + scale_psi + 0.1)
    ax.set_title(f"({potential_type} potential, curvature κ={curvature_factor})")

    # Instead of plotting the initial wavefunction curve, we create an empty line.
    line_psi, = ax.plot([], [], 'b', label=r'Wavefunction $|\psi|^2$')
    line_V, = ax.plot(theta, r_V, 'r', label='Potential')
    ax.legend(loc='upper right')

    # -------------------------------
    # Time Evolution (Animation)
    # -------------------------------
    n_steps = 10000
    frames = 5000

    def update(frame):
        global psi
        # Evolve the wavefunction a few steps per frame
        for _ in range(n_steps // frames):
            psi = U @ psi
            if np.random.uniform(0,1) < 0.01:  # Apply random Reidemeister moves
                psi = apply_reidemeister_sequence(psi, [np.random.choice([R1, R2, R3])])
            psi /= np.linalg.norm(psi)  # re-normalize
        # Update the wavefunction radial profile on the ring
        psi_abs2 = np.abs(psi)**2
        psi_norm = psi_abs2 / np.max(psi_abs2)
        r_psi = R0 + scale_psi * psi_norm
        line_psi.set_data(theta, r_psi)
        return line_psi,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=10, blit=True)
    plt.tight_layout()
    # To save the animation, uncomment the next line (requires ffmpeg)
    ani.save(f"animation_space-time_{potential_type}.mp4", writer="ffmpeg", fps=60)
    #plt.show()