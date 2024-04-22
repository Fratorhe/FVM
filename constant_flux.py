import matplotlib

# matplotlib.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .cell_variable import CellVariable
from .mesh import Mesh
from .model import AdvectionDiffusionModel

if __name__ == "__main__":
    # https://www.mathworks.com/help/symbolic/heating-of-finite-slab.html
    video = True

    # L = 1.0  # m
    # k = 10.0  # W/(m·K)
    # rho = 1000.0  # kg/m³
    # c = 1000.0  # J/(kg·K) # heat capacity
    alpha = 117e-6  # k / (rho * c)  # Thermal diffusivity

    mesh = Mesh.simple_grading(0, 0.6, n_points=50, ratio=0.1)
    print(mesh.faces)

    a = CellVariable(0, mesh=mesh)  # Advection velocity
    d = CellVariable(alpha, mesh=mesh)  # Diffusion coefficient m2/s

    time_step = 0.1e-1  # Time step, s
    total_time = 120  # seconds
    theta = 0  # 1 -> implicit
    left_flux = +3e5 * alpha / 400  # W/m2
    right_flux = 0

    # Initial conditions
    w_init = np.zeros_like(mesh.cells) + 20  # degrees

    model = AdvectionDiffusionModel(
        a,
        d,
        time_step,
        mesh,
        discretization="upwind",
        theta=theta,
    )
    model.set_boundary_conditions(left_flux=left_flux, right_flux=right_flux)

    model.initialize_system(w_init)

    if not video:
        w = model.solve_to_time(total_time)
        fig, ax = plt.subplots()
        ax.plot(mesh.cells, w_init, label="IC")
        ax.plot(mesh.cells, w, label=f"t={total_time}", marker="o")
        plt.show()
    else:
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=15, metadata=metadata)

        fig, ax = plt.subplots()
        l1 = ax.plot([], [], "k-o", markersize=4)[0]
        ann = ax.annotate(f"time 0", xy=(1, 1))

        plt.xlim(np.min(mesh.faces), np.max(mesh.faces))
        plt.ylim(0, 120)
        l1.set_data(mesh.cells, w_init)

        n_steps = int(total_time / time_step)
        with writer.saving(fig, "fixed_heatflux.mp4", 100):
            for i in tqdm(range(n_steps)):
                w = model.take_step()

                if i == 0:
                    l1.set_data(mesh.cells, w_init)
                    writer.grab_frame()

                if i % 100 == 0 or i == 0:
                    l1.set_data(mesh.cells, w)
                    # l0.set_data(analytical_x, analytical_solution2)
                    area = np.sum(w * mesh.cell_widths)

                    if ann is not None:
                        ann.remove()
                    ann = ax.annotate(f"time {i*time_step}", xy=(1, 1))

                    # print("#%d; t=%g; area=%g:" % (i, i * k, area))
                    writer.grab_frame()

        total_time = n_steps * time_step
        print(w)
        print(f"Simulated time: {total_time} s")
