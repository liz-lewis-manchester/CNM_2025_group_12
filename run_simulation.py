import os
import numpy as np

from src.advection import (init_grid, run_simulation, read_ initial_conditions, interpolate_to_grid)


L = 20.0              # domain length (m)
dx = 0.2              # spatial resolution (m)
U = 0.1               # velocity (m/s)
dt = 10.0             # initial timestep (s)
t_total = 300         # simulation time (s) 

os.makedirs("results", exist_ok=True)

#Grid
x, nx = init_grid(L, dx)

# Reading and interpolating initial conditions from csv
df = read_initial_conditions("initial_conditions.csv")
theta0 = interpolate_to_grid(df, x)

# boundary condition at x=0 
boundary_left = lambda t: float(theta0[0])

times, thetas, dt_used, cfl = run_simulation(theta0, U, dx, dt, t_total)

print("Simulation complete")
print(f"- Grid points: {nx}")
print(f"- dt used: {dt_used}")
print(f"- CFL: {cfl}")
print(f"- Time steps: {len(times)}")

np.save("results/times.npy", times)
np.save("thetas.npy", thetas)
print("Saved times.npy and thetas.npy")
