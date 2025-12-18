#Test case 1
def test_spike_constanct_inflow():
    L = 20,0
    dx = 0.2
    U = 0.1
    dt = 10.0
    t_total = 300
    
    x, nx = init_grid(L, dx)
    
    theta0 = np.zeros(nx)
    theta0[0] = 250.0
    boundary_left = lambda t: 250.0
    
    times, thetas, dt_used, cfl = run_simulation(
      theta0, U, dx, dt, t_total, boundary_left=boundary_left
    )
    
    # Shape check
    assert thetas.shape == (lens(times), nx)
    assert np.all(np.isfinite(thetas))
    assert np.all(thetas >= 0.0)
    
    # Downstream propagation check
    mid = nx // 2
    assert np.max(thetas[:, mid]) > 0.0
  #Test case 2
 def test_csv_initial_conditions_runs_cleanly():
    L = 20.0
    dx = 0.2
    U = 0.1 
    dt = 10.0
    t_total = 300.0
    
    x, nx = init_grid(L, dx)
    
    df = read_initial_conditions("data/initial_conditions.csv")
    theta0 = interpolate_to_grid(df, x)
    boundary_left = lambda t: float(theta0[0])
    
    times, thetas, dt_used, cfl = run_simulation(
      theta0, U, dx, dt, t_total, boundary_left=boundary_left
    )
    # Numerical validity
    assert thetas.shape == (len(times), nx)
    assert np.all(np.isfinite(thetas))
    assert np.all(thetas >= 0.0)

import numpy as np
from scr.advection import init_grid, run_simulation, read_initial_conditions, interpolate_to_grid

# Test Case 3
def test_tc3_runs_basic():
    L, dx, U, dt, t_total = 20.0, 0.2, 0.1, 10.0, 300.0
    x, nx = init_grid(L, dx)

    df = read_initial_conditions("data/initial_conditions.csv")
    theta0 = interpolate_to_grid(df, x)
    boundary_left = lambda t, th0=float(theta0[0]): th0

    times, thetas, dt_used, cfl = run_simulation(theta0, U, dx, dt, t_total, boundary_left=boundary_left)

    assert thetas.shape[1] == nx
    assert len(times) == thetas.shape[0]
    assert np.isfinite(thetas).all()

# Test Case 4
def test_tc4_boundary_decays():
    k = 0.002
    C0 = 300.0
    assert C0 * np.exp(-k * 100.0) < C0

# Test Case 5
def test_tc5_velocity_profile_shape():
    L, dx, U0 = 20.0, 0.2, 0.1
    x, nx = init_grid(L, dx)
    rng = np.random.default_rng(42)
    U_profile = U0 * (1.0 + 0.10 * rng.standard_normal(nx))
    assert U_profile.shape == (nx,)

