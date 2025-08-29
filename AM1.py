import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm
from scipy.integrate import quad

# Define spatial and time discretization
dx = 0.1
dt = 0.01
L = 100
x = np.linspace(-L/2, L/2, int(L/dx))
n_x = len(x)

# Define initial wavefunction
def psi_initial(x):
    return np.exp(-x**2) * np.exp(1j * 5 * x)

# Define initial wavefunction
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
    V[idx_center] = V0 / width  # ensuring the area approximates V0
    return V


# (D) Eckart Potential
def eckart_potential(x, V0=10, a=0.2):
    return V0 / (np.cosh(a * x)**2)

# (E) Double Well Potential at Finite Temperature
def double_well_finite_temp_potential(x, a=0.01, b=1, temperature=300, noise_strength=0.1):
    # Base double-well: a(x^2 - b^2)^2
    base = a * (x**2 - b**2)**2
    np.random.seed(42)
    noise = noise_strength * np.random.normal(0, np.sqrt(temperature), len(x))
    return base + noise

# (F) Lennard-Jones Potential (12-6 potential)
def lennard_jones_potential(x, epsilon=1.0, sigma=2.0):
    r = np.abs(x) + 1e-9  # Avoid division by zero
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# (G) 12-6 Potential (identical to Lennard-Jones)
def potential_12_6(x, epsilon=1.0, sigma=2.0):
    r = np.abs(x) + 1e-9
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# (H) Godbeer et al. Potential (Morse-like, shifted)
def godbeer_potential(x, D_e=10, a=0.5, r_e=5):
    # Using absolute value for symmetry and shifting to have minimum -D_e
    return D_e * (1 - np.exp(-a * (np.abs(x) - r_e)))**2 - D_e

# (I) Sitnitsky Potential (candidate double-well form)
def sitnitsky_potential(x, V0=10, a=0.5, b=2):
    # This form uses hyperbolic tangents to generate a double well structure.
    return V0 * (np.tanh(a*(x + b)) - np.tanh(a*(x - b))) / 2



# Potential functions
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

def sine_potential(x, amplitude=1, frequency=0.2):
    return amplitude * np.sin(frequency * x)

# -------------------------------
# Select potential function
# -------------------------------
# Set potential_type to one of:
# "rectangular_barrier", "delta", "finite_square_well", "eckart",
# "double_well_finite_temp", "lennard_jones", "12_6", "godbeer", "sitnitsky"
potential_type = 'double_well_potential'  # Change as desired

ptypes = ["delta","rectangular_barrier", 'quadratic_barrier', 'sine_potential', "double_well_potential",
    "eckart","lennard_jones", "12_6", "godbeer", "sitnitsky"]

for potential_type in ptypes:
    if potential_type == "rectangular_barrier":
        V = rectangular_barrier_potential(x)
    elif potential_type == "delta":
        V = delta_potential(x)
    elif potential_type == "double_well_potential":
        V = double_well_potential(x)
    elif potential_type == "eckart":
        V = eckart_potential(x)
    elif potential_type == "double_well_finite_temp_potential":
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


    # Define Hamiltonian components
    H_free = - np.diag(np.ones(n_x-1), -1) - np.diag(np.ones(n_x-1), 1)
    H_free *= -1 / (2 * dx**2)
    H_potential = np.diag(V)
    H = H_free + H_potential

    # Compute unitary evolution operator
    U = expm(-1j * H * dt)

    # Initialize wavefunction
    psi = psi_initial(x)

    # Define WKB tunneling probability approximation
    def wkb_tunneling_probability(V, E, x_range):
        print(x_range)
        def kappa(x):
            return np.sqrt(2 * max(V[x_range] - E, 0))
        
        x1 = x_range[0]
        x2 = x_range[-1]
        integral, _ = quad(kappa, x1, x2)
        return np.exp(-2 * integral)

    # Compute energy expectation value
    E_expect = np.real(np.vdot(psi, H @ psi))

    # Identify classically forbidden region
    forbidden_region = np.where(V > E_expect)[0]
    if len(forbidden_region) > 0:
        x_tunnel = x[forbidden_region]
        P_WKB = wkb_tunneling_probability(V, E_expect, x_tunnel)
    else:
        P_WKB = 1  # No barrier

    # Plot wavefunction evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    #plt.subplots_adjust(left=None, right=0.2, top=0.2, bottom=0.1)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(0, 0.12)
    ax.set_ylabel(r"$\mathbb{P}$", fontsize = 12)
    ax.set_xlabel(r"$x$", fontsize = 12)
    #ax.set_title("Wavefunction Evolution")

    line, = ax.plot([], [], 'b', label=r'Wavefunction $|\psi|^2$')
    ax.plot(x, V / np.max(V) * 0.1, 'r', label='Potential Barrier')
    ax.legend()

    # Time evolution function
    def update(frame):
        global psi
        psi = U @ psi
        psi /= np.linalg.norm(psi)  
        line.set_data(x, np.abs(psi)**2)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=2000, interval=1, blit=False)
    ani.save(f"animation_qt_wkb_{potential_type}.mp4", writer="ffmpeg", fps=60)
    #plt.show()