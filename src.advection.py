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

  for n in range(1, nt):
    thetas[n] = upwind_step(thetas[n-1], U, dx, dt)
    thetas[n] = apply_boundary(thetas[n])

  return times, thetas, dt, cfl
  
