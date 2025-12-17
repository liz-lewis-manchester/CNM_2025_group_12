import os
import numpy as np
import pandas as pd

def init_grid(L, dx):
  # Create 1D spatial grid with spacing dx from 0 to L.
  nx = int(round(L / dx)) + 1
  x = np.linspace(0.0, float(L), nx)
  return x, nx

def check_cfl(U, dx, dt):
  # CFL number for advection.
  return abs(float(U)) * float(dt) / float(dx)

def upwind_step(theta, U, dx, dt):
  # One explicit upwind step fdr dC/dt + U dC/dx = 0.
  # Assumes U >= 0 (downstream, positive direction).
  U = float(U)
  if U < 0:
    raise ValueError("upwind_step assumes U>=0. For U<0 you need the opposite upwind direction.")
  c = U * float(dt) / float (dx)
  theta_new = theta.copy()

  # interior update (i=1...end).
  theta_new[1:] = theta[1:] - c * (theta[1:] - theta [:-1])
  return theta_new


def apply_boundary_right(theta):
  # At right edge, the outflow boundary is zero-gradient.
  theta[-1] = theta[-2]
  return theta

def run_simulation(theta0, U, dx, dt, t_total, boundary_left=None, enforce_cfl=True, cfl_max=0.9):
  # Run advection simulation.
  # Boundary_left: function boundary_left(t) returning concentration at x=0 for each time t.
  # If None, x=0 is fixed to its initial value.
  theta0 = np.asarray(theta0, dtype=float)
  nx = theta0.size
  
  # CFL handling
  cfl = check_cfl(U, dx, dt)
  if enforce_cfl and (cfl > cfl_max) and (float(U) != 0.0):
    dt = cfl_max * float(dx) / abs(float(U))
    cfl = check_cfl(U, dx, dt)

  nt = int(round(float(t_total) / float(dt))) + 1
  times = np.linspace(0.0, float(dt) * (nt - 1), nt)

  thetas = np.zeros((nt, nx), dtype=float)
  thetas[0] = theta0.copy()

  # Apply left boundary at t=0 too
  if boundary_left is not None:
      thetas[0, 0] = float(boundary_left(times[0]))
  else:
      thetas[0, 0] = thetas[0, 0]
        
  for n in range(1, nt):
    thetas[n] = upwind_step(thetas[n-1], U, dx, dt)

    # Left boundary (x=0 inflow)
    if boundary_left is not None:
        thetas[n, 0] = float(boundary_left(times[n]))
    else:
        thetas[n, 0] = thetas[0, 0] # keep initial value

  thetas[n] = apply_boundary_right(thetas[n])
  thetas[n] = np.maximum(thetas[n], 0.0)
        
  return times, thetas, dt, cfl

def read_initial_conditions(csv_path):
  # Reads initial_conditions.csv with ANY two-cloumn header.
  # Initial conditions: Distance(m), Concentration(µg/m_).
  # We will always take the first two columns and rename them to x, theta.
  df = pd.read_csv(csv_path, encoding="unicode_escape")

  # Always take the first two columns and rename.
  df = df.iloc[:, :2].copy()
  df.columns = ["x", "theta"]

  df["x"] = df["x"].astype(float)
  df["theta"] = df["theta"].astype(float)

  df = df.sort_values("x").reset_index(drop=True)
  return df

def interpolate_to_grid(df, x_grid):
  # Linear interpolation onto grid using NumPy (no Scipy dependency).
  # Outside the measured range, fill with 0.
  x_meas = df["x"].to_numpy(dtype=float)
  theta_meas = df["theta"].to_numpy(dtype=float)

  theta0 = np.interp(x_grid, x_meas, theta_meas, left=0.0, right=0.0)
  theta0 = np.maximum(theta0, 0.0)
  return theta0
  
  
