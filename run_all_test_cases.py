import os
import numpy as np

from scr.advection import init_grid, run simulation, read_initial_conditions, interporlate_to_grid

def save_case(case_id, times,  theta, meta):
  os.makedirs("results", exist_ok=True)

np.save(f"results/{case_id}_times.npy", times)
np.save(f"results/{case_id}_thetas.npy", thetas)

with open(f"results/{case_id}_meta.txt", "w", encoding="uft-8") as f:
  for k, v in meta.items():
    f.write(f"{k}: {v}/n")

def main():
  os.makedirs("results", exist_ok=True)

#Parameters for test cases
L = 20.0
dx = 0.2
U = 0.1 
dt = 10.0 
t_total = 300.0

x, nx = init_grid(L, dx)

#Test case 1
theta0_1 = np.zeros(nx)
theta0_1[0] = 250.0
boundary_1 = lambda t: 250.0

times1, thetas1, dt1, cfl1 = run_simulation(theta0_1, U, dx, dt, t_total, boundary_left=boundary_1)
save_case("tc1_spike_constant_inflow", times1, thetas1,
          {"L": L, "dx": dx, "U": U, "dt_used": dt1, "t_total": t_total, "CFL": cfl1})

#Test case 2
# Read initial conditions from the csv and interpolate to grid
df = read_initial_conditions("data/initial_conditions.csv")
theta0_2 = interpolate_to_grid(df, x)
boundary_2 = lambda t: float(theta0_2[0])

times2, thetas2, dt2, cfl2 = run_simulation(theta0_2, U, dx, dt, t_total, boundary_left= boundary_2)
save_case("tcs2_csv_initial_conditions", times2, thetas2, 
          {"L": L, "dx": dx, "U": U, "dt_used": dt2, "t_total": t_total, "CFL": cfl2})
