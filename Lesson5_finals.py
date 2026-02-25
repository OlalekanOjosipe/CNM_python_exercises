import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================
# Exercise 1 – Monte Carlo pi
# ============================================

def estimate_pi(N=10000, seed=None, plot=False):
    
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N)
    y = rng.uniform(-1, 1, N)

    inside = (x**2 + y**2) <= 1
    inside_count = np.sum(inside)

    pi_est = 4 * inside_count / N

    if plot:
        theta = np.linspace(0, 2*np.pi, 400)
        cx = np.cos(theta)
        cy = np.sin(theta)

        plt.figure(figsize=(6, 6))
        # square border
        plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1])
        # circle border
        plt.plot(cx, cy)

        plt.scatter(x[inside], y[inside], s=5, alpha=0.6, label="Inside")
        plt.scatter(x[~inside], y[~inside], s=5, alpha=0.6, label="Outside")

        plt.title(f"Monte Carlo Pi Estimate (N={N}) ≈ {pi_est:.6f}")
        plt.axis("equal")
        plt.legend()
        plt.show()

    return pi_est, inside_count

# test example
pi_est, inside = estimate_pi(N=20000, seed=0, plot=True)
print("Estimated pi:", pi_est)


# ============================================
# Exercise 2 – Bessel Function of First Kind
# ============================================

import numpy as np
import math
import matplotlib.pyplot as plt


def bessel_J(n, x, tol=1e-12, max_terms=200):
    """
    Compute the Bessel function of the first kind J_n(x)
    using its power series expansion.

    Parameters
    ----------
    n : int
        Order of the Bessel function (n >= 0).
    x : float or numpy array
        Non-negative argument(s).
    tol : float
        Tolerance for stopping the series.
    max_terms : int
        Maximum number of terms in the series.

    Returns
    -------
    J : float or numpy array
        Approximation of J_n(x).
    """

    if n < 0 or int(n) != n:
        raise ValueError("Order n must be a non-negative integer.")

    x = np.asarray(x, dtype=float)

    if np.any(x < 0):
        raise ValueError("x must be non-negative.")

    J = np.zeros_like(x)

    for m in range(max_terms):
        term = ((-1)**m) * (x/2)**(2*m + n) / \
               (math.factorial(m) * math.factorial(m + n))

        J += term

        if np.max(np.abs(term)) < tol:
            break

    return J


# -------------------------------------------------
# Plot J_0(x) to J_5(x) for x in [0,10]
# -------------------------------------------------

x_vals = np.linspace(0, 10, 400)

plt.figure(figsize=(8, 6))
for n in range(0, 6):
    plt.plot(x_vals, bessel_J(n, x_vals), label=f"$J_{n}(x)$")

plt.title("Bessel Functions of the First Kind")
plt.xlabel("x")
plt.ylabel("$J_n(x)$")
plt.legend()
plt.grid(True)
plt.show()


# =============================
# Exercise 3 random walk
# =============================

# Parameters (change these)
# ----------------------------
Nstep = 200   # steps per walk
Nw    = 5000  # number of walks
Nsamp = 10    # how many trajectories to plot as a sample

rng = np.random.default_rng(0)

# ---------------------------------------------------------
# 1) Generate random steps in x and y: each in {-1, 0, +1}
# ---------------------------------------------------------
dx = rng.integers(-1, 2, size=(Nw, Nstep))  # -1,0,1
dy = rng.integers(-1, 2, size=(Nw, Nstep))  # -1,0,1

# ---------------------------------------------------------
# Build trajectories using cumulative sums
# x, y have shape (Nw, Nstep+1), starting at 0
# ---------------------------------------------------------
x = np.hstack([np.zeros((Nw, 1), dtype=int), np.cumsum(dx, axis=1)])
y = np.hstack([np.zeros((Nw, 1), dtype=int), np.cumsum(dy, axis=1)])

# Final points (shape (Nw,))
x_end = x[:, -1]
y_end = y[:, -1]

# ---------------------------------------------------------
# 4) Distance of final points from origin + average distance
# ---------------------------------------------------------
r = np.sqrt(x_end**2 + y_end**2)
avg_r = np.mean(r)
print("Average final distance from origin:", avg_r)

# ---------------------------------------------------------
# 2) Plot a sample of trajectories
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
for i in range(min(Nsamp, Nw)):
    plt.plot(x[i, :], y[i, :], linewidth=1)
plt.scatter([0], [0], marker="x")  # origin
plt.title(f"Sample of {min(Nsamp, Nw)} random-walk trajectories (Nstep={Nstep})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 3) Plot all final points together
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(x_end, y_end, s=5, alpha=0.4)
plt.scatter([0], [0], marker="x")
plt.title(f"Final points of {Nw} random walks")
plt.xlabel("x_end")
plt.ylabel("y_end")
plt.axis("equal")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 5) Histogram of distances to origin
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(r, bins=40)
plt.title("Histogram of final distances from origin")
plt.xlabel("distance r")
plt.ylabel("count")
plt.grid(True)
plt.show()



# ================================================
# Exercise 4 - Julia's set
#=================================================
# Parameters
a_values = [(-0.5, 0), (0.25, -0.52), (-1, 0), (-0.2, 0.66)]
max_iter = 100
grid_size = 1000

# Grid definition
real_axis = np.linspace(-1, 1, grid_size)
imag_axis = np.linspace(-1, 1, grid_size)
Re, Im = np.meshgrid(real_axis, imag_axis)


def julia_set(a, Re, Im, max_iter):
    # Current iteration values (start at z0 = Re + i Im)
    x_current = Re.copy()
    y_current = Im.copy()

    # Mask: True = still in set
    inside = np.ones(Re.shape, dtype=bool)

    a_real, a_imag = a

    # Escape radius
    escape_radius = max(2.0, np.sqrt(a_real**2 + a_imag**2))
    escape_radius_sq = escape_radius**2

    for _ in range(max_iter):
        # z^2 + a
        x_next = x_current**2 - y_current**2 + a_real
        y_next = 2*x_current*y_current + a_imag

        x_current = x_next
        y_current = y_next

        # Check escape
        escaped = (x_current**2 + y_current**2) > escape_radius_sq
        inside[escaped] = False

        # Optional: stop updating escaped points
        x_current[escaped] = 0
        y_current[escaped] = 0

    return inside


# Plot results
for a in a_values:
    plt.figure(dpi=300)
    mask = julia_set(a, Re, Im, max_iter)
    plt.imshow(mask, extent=(-1, 1, -1, 1), origin="lower", cmap="terrain")
    plt.title(f"Filled Julia Set for a = {a}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.show()