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
