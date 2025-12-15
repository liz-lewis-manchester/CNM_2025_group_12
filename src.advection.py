import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def init_grid(L, dx):
  nx = int(round(L / dx)) + 1
  x = np.linspace(0, L, nx)
  return x, nx

def check_cfl(U, dx, dt):
  return abs(U) * dt/dx

def upwind_step(theta, U, dx, dt):
  c = U * dt/dx
  theta_new = theta.copy()
  theta_new[1:] = theta[1:] - c * (theta[1:] - theta[:-1])
  return theta_new

def apply_boundary(theta):
  theta[-1] = theta[-2]
  return theta

def run_simulation(theta0, U, dx, dt, t_total, enforce_cfl=True, cfl_max=0.9):
  nx = theta0.size
  cfl = check_cfl(U, dx, dt)

  if enforce_cfl and cfl > cfl_max and U !=0:
    dt = cfl_max * dx / abs(U)
    cfl = check_cfl(U, dx, dt)

  nt = int(round(t_total / dt)) + 1
  times = np.linspace(0, dt * (nt - 1), nt)

  thetas = np.zeros((nt, nx))
  thetas[0] = theta0.copy()

  # Left boundary (x=0) remains fixed from the initial condition
  # Right boundary handled by apply_boundary (zero-gradient outflow)

  for n in range(1, nt):
    thetas[n] = upwind_step(thetas[n-1], U, dx, dt)
    thetas[n] = apply_boundary(thetas[n])

  return times, thetas, dt, cfl

def read_initial_conditions(csv_path):
  df = pd.read_csv(csv_path, encoding="unicode_escape")
  if 'x' not in df.columns or 'theta' not in df.columns:
    df = df.iloc[:, :2]
    df.columns = ['x', 'theta']
  df = df.sort_values('x').reset_index(drop=True)
  return df

def interpolate_to_grid(df, x_grid, kind= 'linear', fill_value =0.0):
  f  = interp1d(df['x'].values, 
             df['theta'].values,
             kind=kind,
             bounds_error=False,
             fill_value=fill_value)
  theta0 = f(x_grid)
  return theta0 
  
  
