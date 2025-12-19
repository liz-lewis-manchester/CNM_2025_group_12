import os
import numpy as np

from src.advection import init_grid, run_simulation, read_initial_conditions, interporlate_to_grid, upwind_step, apply_boundary_right

def save_case(case_id, times, thetas, meta):
  os.makedirs("results", exist_ok=True)
  
  np.save(f"results/{case_id}_times.npy", times)
  np.save(f"results/{case_id}_thetas.npy", thetas)
  with open(f"results/{case_id}_meta.txt", "w", encoding="utf-8") as f:
    for k, v in meta.items():
      f.write(f"{k}: {v}\n")

def main():
  os.makedirs("results", exist_ok=True)

  # Parameters for test cases
  L = 20.0
  dx = 0.2
  U = 0.1
  dt = 10.0
  t_total = 300.0

  x, nx = init_grid(L, dx)
 
  # Test case 1
  theta0_1 = np.zeros(nx)
  theta0_1[0] = 250.0
  boundary_1 = lambda t: 250.0

  times1, thetas1, dt1, cfl1 = run_simulation(
    theta0_1, U, dx, dt, t_total, boundary_left=boundary_1
  )

  save_case(
    "tc1_spike_constant_inflow",
    times1, thetas1,
    {"L": L, "dx": dx, "U": U, "dt_used": dt1, "t_total": t_total, "CFL": cfl1}
  )

  # Test case 2
  # Read initial conditions from the csv and interpolate to grid
  df = read_initial_conditions("data/initial_conditions.csv")
  theta0_2 = interpolate_to_grid(df, x)
  boundary_2 = lambda t: float(theta0_2[0])

  times2, thetas2, dt2, cfl2 = run_simulation(
    theta0_2, U, dx, dt, t_total, boundary_left=boundary_2
  )

  save_case(
    "tc2_csv_initial_conditions",
    times2, thetas2,
    {"L": L, "dx": dx, "U": U, "dt_used": dt2, "t_total": t_total, "CFL": cfl2}
  )

# Test Case 3
def run_tc3(L=20.0, t_total=300.0):
    os.makedirs("results", exist_ok=True)

    U_list = [0.05, 0.1, 0.2]
    dx_list = [0.1, 0.2, 0.5]
    dt_list = [5.0, 10.0, 20.0]

    df_ic = read_initial_conditions("data/initial_conditions.csv")

    case_idx = 0
    for dx in dx_list:
        x, nx = init_grid(L, dx)
        theta0 = interpolate_to_grid(df_ic, x)
        boundary_left = lambda t, th0=float(theta0[0]): th0

        for U in U_list:
            for dt in dt_list:
                case_idx += 1
                times, thetas, dt_used, cfl = run_simulation(
                    theta0, U, dx, dt, t_total, boundary_left=boundary_left
                )
                save_case(
                    f"tc3_{case_idx:02d}_U{U}_dx{dx}_dt{dt}".replace(".", "p"),
                    times, thetas,
                    {"L": L, "dx": dx, "U": U, "dt_requested": dt, "dt_used": dt_used, "t_total": t_total, "CFL": cfl}
                )


# Test Case 4
def run_tc4(L=20.0, dx=0.2, U=0.1, dt=10.0, t_total=300.0, k=0.002):
    x, nx = init_grid(L, dx)

    df_ic = read_initial_conditions("data/initial_conditions.csv")
    theta0 = interpolate_to_grid(df_ic, x)

    C0 = float(theta0[0])
    boundary_left = lambda t: C0 * np.exp(-k * t)

    times, thetas, dt_used, cfl = run_simulation(theta0, U, dx, dt, t_total, boundary_left=boundary_left)

    save_case(
        "tc4_exponential_decay_boundary",
        times, thetas,
        {"L": L, "dx": dx, "U": U, "dt_used": dt_used, "t_total": t_total, "CFL": cfl, "C0": C0, "k": k}
    )


# Test Case 5
def run_tc5(L=20.0, dx=0.2, U0=0.1, dt=10.0, t_total=300.0, seed=42):
    x, nx = init_grid(L, dx)

    df_ic = read_initial_conditions("data/initial_conditions.csv")
    theta0 = interpolate_to_grid(df_ic, x)

    rng = np.random.default_rng(seed)
    U_profile = U0 * (1.0 + 0.10 * rng.standard_normal(nx))

    boundary_left = lambda t, th0=float(theta0[0]): th0

    # NOTE: this requires run_simulation to accept U as an array.
    times, thetas, dt_used, cfl = run_simulation(theta0, U_profile, dx, dt, t_total, boundary_left=boundary_left)

    save_case(
        "tc5_variable_velocity_10pct",
        times, thetas,
        {"L": L, "dx": dx, "U0": U0, "dt_used": dt_used, "t_total": t_total, "CFL": cfl, "seed": seed}
    )


def main():
    run_tc3()
    run_tc4()
    run_tc5()


if __name__ == "__main__":
    main()

  
  

  

  





  
  



