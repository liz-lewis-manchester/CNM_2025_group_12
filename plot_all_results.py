import os
import numpy as np
import matplotlib.pyplot as plt

from src.advection import init_grid

#plotting the utiilities
def plot_final(x, thetas, title, outpath):
  plt.figure()
  plt.plot(x, thetas[-1])
  plt.xlabel("Distance x (m)")
  plt.ylabel("Concentration")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def plot_heatmap(x, times, thetas, title, outpath):
  plt.figure()
  plt.imshow(
    thetas,
    aspect="auto",
    origin="lower",
    extent=[x.min(), x.max(), times.min(), times.max()]
  )
  plt.xlabel("Distance x (m)")
  plt.ylabel("Time t (s)")
  plt.title(title)
  plt.colorbar(label="Concentration")
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def main():
  os.makedirs("results", exist_ok=True)

  # default domain
  L = 20.0
  dx_default = 0.2
  # loop through all saved cases
  for fname in os.listdir("results"):
    if fname.endswith("_thetas.npy"):
      case_id = fname.replace("_thetas.npy", "")
      thetas = np.load(f"results/{case_id}_thetas.npy")
      times = np.load(f"results/{case_id}_times.npy")

      dx = dx_default
      meta_path = f"results/{case_id}_meta.txt"
      if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
          for line in f:
            if line.startswith("dx:"):
              dx = float(line.split(":")[1].strip())

      x, nx = init_grid(L, dx)
      if thetas.shape[1] != nx:
        print(f"Skipping {case_id} (grid mismatch)")
        continue

      #generate plots
      plot_final(x, thetas, f"{case_id} final snapshot", f"results/{case_id}_final.png")
      plot_heatmap(x, times, thetas, f"{case_id} heatmap", f"results/{case_id}_heatmap.png")

  print("Plotting complete.")


if __name__ == "__main__":
  main()
