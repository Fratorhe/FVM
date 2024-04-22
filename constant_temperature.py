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
    video = False

    mesh = Mesh.uniform(-1, 1, n_points=10)
    print(mesh.faces)

    a = CellVariable(0, mesh=mesh)  # Advection velocity
    d = CellVariable(1, mesh=mesh)  # Diffusion coefficient m2/s

    time_step = 1e-3  # Time step, s
    total_time = 1  # seconds
    theta = 1
    left_value = 1
    right_value = 1

    # Initial conditions
    w_init = np.zeros_like(mesh.cells)  # degrees
    w_init[0] = left_value  # degrees
    w_init[-1] = right_value  # degrees

    model = AdvectionDiffusionModel(
        a, d, time_step, mesh, discretization="upwind", theta=theta
    )
    model.set_boundary_conditions(left_value=left_value, right_value=right_value)
    # model.set_boundary_conditions(left_flux=left_flux, right_flux=left_flux)

    model.initialize_system(w_init)

    if not video:
        w = model.solve_to_time(total_time)
        fig, ax = plt.subplots()
        ax.plot(mesh.cells, w_init, label="IC")
        ax.plot(mesh.cells, w, label=f"t={total_time}")
        plt.show()

    else:
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
        writer = FFMpegWriter(fps=15, metadata=metadata)

        fig, ax = plt.subplots()
        l1 = ax.plot([], [], "k-o", markersize=4)[0]
        ann = ax.annotate(f"time 0", xy=(1, 1))

        plt.xlim(np.min(mesh.faces), np.max(mesh.faces))
        plt.ylim(0, 1)
        l1.set_data(mesh.cells, w_init)

        n_steps = int(total_time / time_step)
        with writer.saving(fig, "fixed_temperatures.mp4", 100):
            for i in tqdm(range(n_steps)):
                w = model.take_step()

                if i == 0:
                    l1.set_data(mesh.cells, w_init)
                    writer.grab_frame()

                if i % 1 == 0 or i == 0:
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
