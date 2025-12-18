# CNM_2025_group_12

# Model of Pollutant Transport in a River

## Task Description 
This coursework involves developing a Python-based numerical model to simulate the transport of pollutant in a river over time and space.

The objective was to implement a 1-dimensional advection model that computes pollutant concentration as a function of distance and time downstream. 
The model solves the advection equation using finite difference methods in both the spatial and temporal domains. 

The project integrates programming, numerical modelling, data handling, testing, visualisation and collaborative software development using GitHub. 

--- 

## Mathematical Model
Pollutant transport modelled using the 1-dimensional advection equation:

dC/dt + U dC/dx = 0 
where:
- C is pollutant concentration (μg/m^3)
- t is time (s)
- x is distance along the river (m)
- U is stream velocity (ms^-1)

The equation is discretised with the finite difference methods in time and space for numerical simulation. 

---

## Repository Structure
Our task's repository follows this structure

```
├── src/
├── tests/
├── README.md
├── initial_conditions.csv
├── plot_all_results.py
├── run_all_test_cases.py
└── run_simulation.py
```
---

## Installation 
Ensure that Python is installed.

---

## Running the model 
To run a single simulation:
```bash
python_run_all_test_cases.py
```

to run all test cases:
```bash
python run_all_test_cases.py
```
The model allows user to specify domain, spatial and temporal resolution and stream velocity.

Initial conditons are read from 'initial_conditions.csv' and interpolated onto the model.

---

## Test Cases 
The following test cases were implemented:
1. Simulation of pollutant transpoprt over a 20m river domain for 5 minutes with constant velocity and fixed initial concentration.
2. Simulation using initial conditions read from 'initial_conditions.csv' and interpolated onto the model grid.
3. Sensitivity testing of model parameters, including stream velocity, spatial and temporal resolution.
4. Simulation using a variable stream velocity profile.

Automated tests are implemented in the 'tests' directory using 'pytest'.

To run all automated tests:
```bash
pytest
```

---

## Version Control and Branching 
GitHub was utilised for all version control and collaboration.

- The `main` branch contains stab;e and reviewed code.
- The `model-core` branch was used for developemt of the core numerical model.
- The `Task-1` branch was used for implementing specific tasks.

All changes were merged into `main` using pull requests, with team members reviewing each-others contributions. 

---

## Contribution Breakdown 
This project was completed as a group.
- Core numerical model development was carried out in the `model-core` branch.
- Test case implementation and validation were developed on specific task branches.
- Plotting and integration were handles separately to ensure clarity.
- README.md was developed throughout the process.

---
## 
- Student names
  - Amna Farooq (11577901)
  - x
  - x
  - x
  - x
- Civil Engineering (CIVL20471)





