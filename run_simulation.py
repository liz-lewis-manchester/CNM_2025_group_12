import numpy as np
from src.advection import (init_grid, check_cfl, upwind_step, apply_boundary, run_simulation)


L = 20.0              # domain length
dx = 0.2              # spatial resolution
U = 0.1               # velocity
dt = 10.0             # initial timestep
t_total = 300         # simulation time (5 minutes)

x, nx = init_grid(L, dx)

df = read_initial_conditions("initial_conditions.csv")

theta0 = interpolate_to_grid(df, x)

times, thetas, dt_used, cfl = run_simulation(theta0, U, dx, dt, t_total)

print("Simulation complete")
print(f"- Grid points: {nx}")
print(f"- dt used: {dt_used}")
print(f"- CFL: {CFL}")
print(f"- Time steps: {len(times)}")

np.save("results/times.npy", times)
np.save("thetas.npy", thetas)
print("Saved times.npy and thetas.npy")
