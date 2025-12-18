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
def test_refinement_consistency(csv_path):
  """
  Sensitivity to spatial/temporal resolution:
  compare coarse vs finer run (fine interpolated onto coarse grid).
  """
  L, t_total, U = 20.0, 300.0, 0.1
  df = read_initial_condition(csv_path)
  
  # coarse
  dx1, dt1 = 0.2, 10.0
  x1, _ = init_grid(L, dx1)
  th1 = interpolate_to_grid(df, x1)
  b1 = lambda t: float(th1[0])
  _, C1, _, _ = run_simulation(th2, U, dx2, dt2, t_total, boundary_left=b2)

  # compare final profiles (fine -> coarse)
  C2_final_on_x1 = np.interp(x1, x2, C2[-1])
  rmse = np.sqrt(np.mean((C1[-1] - C2_final_on_x1) ** 2))
  
  # upwind is diffusive; allow some difference, but not huge
  assert rmse < 50.0

def test_3_velocity_arrives_earlier(csv_path):
  """
  Sensitivity to velocity U:
  faster U should make pollutant reach a downstream probe earlier.
  """
  L, t_total, dx, dt = 20.0, 300.0, 0.2, 10.0
  df = read_initial_conditions(csv_path)
  x, _= init_grid(L, dx)
  th = interpolate_to_grid(df, x)
  b = lambda t: float(th[0])
  
  _, Cslow, _, _ = run_simulation(th, 0.05, dx, dt, t_total, boundary_left=b)
  _, Cfast, _, _ = run_simulation(th, 0.20, dx, dt, t_total, boundary_left=b)
  
  i = int(round(10.0 / dx))  # 10 m probe
  thr = 1.0
  
  def first_hit(col):
    hits = np.where(col > thr)[0]
    return hits[0] if len(hits) else None
  
  ts = first_hit(Cslow[:, i])
  tf = first_hit(Cslow[:, i])
  
  assert ts is not None and tf is not None
  assert tf < ts

# Test Case 4

def test_4_exponential_decay_reduces_peak(csv_path):
  """
  Exponentially decaying boundary inflow:
  C_in(t) = C0 * exp(-t/tau) should reduce downstream peak vs constant inflow.
  """
  L, t_total, dx, dt, U = 20.0, 300.0, 0.2, 10.0, 0.1
  df = read_initial_conditions(csv_path)
  x, _ = init_grid(L, dx)
  th = interpolate_to_grid(df, x)
  
  C0 = 250.0
  tau = 60.0 # seconds
  
  boundary_const = lambda t: C0
  boundary_decay = lambda t: C0 * np.exp(-t / tau)
  
  _, C_const, _, _ = run_simulation(th, U, dx, dt, t_total, boundary_left=boundary_const)
  _, C_decay, _, _ = run_simulation(th, U, dx, dt, t_total, boundary_left=boundary_decay)
  
  i = int(round(10.0 / dx))
  assert C_decay[:, i].max() < C_const[:, i].max()

# Test Case 5

def run_time_varying_U(theta0, U_series, dx, dt, t_total, boundary_left=None):
  theta0 = np.asarray(theta0, float)
  nx = theta0.size
  nt = int(round(float(t_total) / float(dt))) + 1
  times = np.linspace(0.0, float(dt) * (nt - 1), nt)

  thetas = np.zeros((nt, nx), float)
  thetas[0] = theta0.copy()
  if boundary_left is not None:
    thetas[0, 0] = float(boundary_left(times[0]))
    
  for n in range(1, nt):
    thetas[n] = upwind_step(theta[n-1], float(U_series[n]), dx, dt)
    if boundary_left is not None:
      thetas[n, 0] = float(boundary_left(times[n]))
    else:
      thetas[n,0] = thetas[0,0]
    thetas[n] = apply_boundary_right(thetas[n])
    thetas[n] = np.maximum(thetas[n], 0.0)
    
  return times, thetas

def test_5_variable_velocity_10pct_reproducible_and_changes_solution(csv_path):
  """
  Variable velocity test with 10% random perturbation:
  - reproducible with fixed seed
  - different from constant-velocity baseline
  """
  L, t_total, dx, dt, U0 = 20.0, 300.0, 0.2, 10.0, 0.1
  df = read_initial_conditions(csv_path)
  x, nx = init_grid(L, dx)
  th = interpolate_to_grid(df, x)
  boundary_left = lambda t: float(th[0])
  
  # baseline constant U  
  _, C_base, _, _ = run_simulation(th, U0, dx, dt, t_total, boundary_left=boundary_left)
  
  rng1 = np.random.default_rng(123)
  
  # Try spatial velocity profile U(x) first (preferred interpretation of "velocity profile")
  U_prof1 = np.maximum(1e-6, U0 * (1.0 + 0.10 * rng1.standard_normal(nx)))
  rng2 = np.random.default_rng(123)
  U_prof2 = np.maximum(1e-6, U0 * (1.0 + 0.10 * rng2.standard_normal(nx)))
  
  used_profile = True
  try:
      _, C_var1, _, _ = run_simulation(th, U_prof1, dx, dt, t_total, boundary_left=boundary_left)
      _, C_var2, _, _ = run_simulation(th, U_prof1, dx, dt, t_total, boundary_left=boundary_left)
  except Exception:
    used_profile = False
    # Fallback: time-varying scalar U(t)
    nt = int(round(t_total / ct)) + 1
    rng1 = np.random.default_rng(123)
    U_series1 = np.maximum(1e-6, U0 * (1.0 + 0.10 * rng1.standard_normal(nt)))
    rng2 = np.random.default_rng(123)
    U_series2 = np.maximum(1e-6, U0 * (1.0 + 0.10 * rng2.standard_normal(nt)))
    
    _, C_var1 = run_time_varying_U(th, U_series1, dx, dt, t_total, boundary_left=boundary_left)
    _, C_var2 = run_time_varying_U(th, U_series2, dx, dt, t_total, boundary_left=boundary_left)
  
  # same seed => identical outputs
  assert np.allclose(C_var1, C_var2)
  ff
  # must differ from constant-U baseline
  mean_abd_diff = float(np.mean(np.abs(C_var1 - C_base)))
  assert mean_abs_diff > 1e-3

if __name__ == "__main__":
  main()
  
  

  

  





  
  



