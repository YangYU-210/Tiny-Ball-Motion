# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:07:48 2025

@author: yy100
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation

# Parameters
grid_size = 200
domain_size = 25
num_obstacles = 5
obstacle_size = 20
num_particles = 50
dt = 0.01
T = 20
N = int(T / dt)
r = 0.01
m = 1.0

# Seed for reproducibility
np.random.seed(321)

# 1. Generate viscosity grid
raw_viscosity = np.random.rand(grid_size, grid_size)
viscosity_grid = gaussian_filter(raw_viscosity, sigma=8)
viscosity_grid = (viscosity_grid - viscosity_grid.min()) / (viscosity_grid.max() - viscosity_grid.min()) * 2.5 + 0.5

# 2. Generate obstacles and record physical bounds (using pixel edge alignment)
obstacle_grid = np.zeros((grid_size, grid_size))
obstacle_bounds = []

def grid_to_phys_edge(i, grid_size, domain_size):
    return i / grid_size * domain_size - domain_size / 2

for _ in range(num_obstacles):
    x_start = np.random.randint(10, grid_size - obstacle_size - 10)
    y_start = np.random.randint(10, grid_size - obstacle_size - 10)
    obstacle_grid[x_start:x_start + obstacle_size, y_start:y_start + obstacle_size] = 1

    x0 = grid_to_phys_edge(x_start, grid_size, domain_size)
    x1 = grid_to_phys_edge(x_start + obstacle_size, grid_size, domain_size)
    y0 = grid_to_phys_edge(y_start, grid_size, domain_size)
    y1 = grid_to_phys_edge(y_start + obstacle_size, grid_size, domain_size)
    obstacle_bounds.append((x0, x1, y0, y1))

# 3. Mask obstacle region in all physical quantities
#viscosity_grid[obstacle_grid == 1] = 1e6
D_grid = 1.0 / viscosity_grid
D_grid[obstacle_grid == 1] = 0.0
force_grid_x = np.random.uniform(-0.05, 0.05, (grid_size, grid_size))
force_grid_y = np.random.uniform(-0.05, 0.05, (grid_size, grid_size))
force_grid_x[obstacle_grid == 1] = 0.0
force_grid_y[obstacle_grid == 1] = 0.0

# 4. Initialize particle arrays
x = np.zeros((num_particles, N))
y = np.zeros((num_particles, N))
vx = np.zeros((num_particles, N))
vy = np.zeros((num_particles, N))
eta_x = np.random.normal(0, 1, (num_particles, N))
eta_y = np.random.normal(0, 1, (num_particles, N))

# Define functions
def grid_to_phys_center(g, grid_size, domain_size):
    return (g + 0.5) / grid_size * domain_size - domain_size / 2

def is_in_obstacle(x, y, bounds, margin=0.01):
    for (x0, x1, y0, y1) in bounds:
        if (x0 - margin <= x <= x1 + margin) and (y0 - margin <= y <= y1 + margin):
            return True
    return False

# Initialize particle positions around a source (0,0)
x0, y0 = 0.0, 0.0  # Release center
initial_radius = 1.0  # Initial dispersion radius

for i in range(num_particles):
    while True:
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, initial_radius)
        xi = x0 + radius * np.cos(angle)
        yi = y0 + radius * np.sin(angle)
        if (-domain_size/2 <= xi <= domain_size/2 and
            -domain_size/2 <= yi <= domain_size/2 and
            not is_in_obstacle(xi, yi, obstacle_bounds, margin=r)):
            x[i, 0] = xi
            y[i, 0] = yi
            break

# 5. Langevin simulation with physical-bound obstacle rejection
for n in range(N - 1):
    for i in range(num_particles):
        gx = int((x[i, n] + domain_size / 2) / domain_size * grid_size)
        gy = int((y[i, n] + domain_size / 2) / domain_size * grid_size)
        gx = np.clip(gx, 0, grid_size - 1)
        gy = np.clip(gy, 0, grid_size - 1)

        D = D_grid[gx, gy]
        beta = (6 * np.pi * viscosity_grid[gx, gy] * r) / m
        Fx = force_grid_x[gx, gy]
        Fy = force_grid_y[gx, gy]

        vx_new = vx[i, n] - beta * vx[i, n] * dt + np.sqrt(2 * D * dt) * eta_x[i, n] + Fx * dt / m
        vy_new = vy[i, n] - beta * vy[i, n] * dt + np.sqrt(2 * D * dt) * eta_y[i, n] + Fy * dt / m

        x_new = x[i, n] + vx_new * dt
        y_new = y[i, n] + vy_new * dt

        if (-domain_size/2 <= x_new <= domain_size/2 and
            -domain_size/2 <= y_new <= domain_size/2 and
            not is_in_obstacle(x_new, y_new, obstacle_bounds, margin=r)):
            x[i, n+1] = x_new
            y[i, n+1] = y_new
            vx[i, n+1] = vx_new
            vy[i, n+1] = vy_new
        else:
            x[i, n+1] = x[i, n]
            y[i, n+1] = y[i, n]
            vx[i, n+1] = -vx[i, n] * 0.3 + np.random.normal(0, 0.01)
            vy[i, n+1] = -vy[i, n] * 0.3 + np.random.normal(0, 0.01)

# 6. Create MP4 animation
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(viscosity_grid, extent=[-12.5, 12.5, -12.5, 12.5], origin="lower", cmap="coolwarm", alpha=0.7)
#ax.imshow(obstacle_grid, extent=[-12.5, 12.5, -12.5, 12.5], origin="lower", cmap="gray", alpha=0.5)
for (x0_bound, x1_bound, y0_bound, y1_bound) in obstacle_bounds:
    ax.plot([x0_bound, x1_bound, x1_bound, x0_bound, x0_bound], [y0_bound, y0_bound, y1_bound, y1_bound, y0_bound], 'k-', lw=2)

particle_dots = [ax.plot([], [], 'o', markersize=3)[0] for _ in range(num_particles)]
ax.set_xlim(-12.5, 12.5)
ax.set_ylim(-12.5, 12.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Particle Motion Animation")
ax.grid(True)

# Animation function
def update(frame):
    for i, dot in enumerate(particle_dots):
        dot.set_data(x[i, frame], y[i, frame])
    return particle_dots

step = 3
frames = range(0, N, step)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
ani.save("particle_motion.mp4", writer="ffmpeg", fps=30)
plt.close()

print("MP4 animation saved as 'particle_motion.mp4'.")
