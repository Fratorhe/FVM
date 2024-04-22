import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import dia_matrix, identity
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from .aux_funcs import CFL_condition, peclet_number
from .bcs import (
    dirichlet_boundary_condition_matrix_elements_left,
    dirichlet_boundary_condition_matrix_elements_right,
    dirichlet_boundary_condition_vector_elements_left,
    dirichlet_boundary_condition_vector_elements_right,
    robin_boundary_condition_matrix_elements_left,
    robin_boundary_condition_matrix_elements_right,
    robin_boundary_condition_vector_elements_left,
    robin_boundary_condition_vector_elements_right,
)
from .cell_variable import CellVariable
from .mesh import Mesh


class AdvectionDiffusionModel:
    """A model for the advection-diffusion equation"""

    def __init__(
        self,
        a,
        d,
        time_step,
        mesh: Mesh,
        discretization: str = "central",
        theta: float = 0.5,
        s=0,
    ):
        self.mesh = mesh
        self.a = CellVariable(a, mesh=self.mesh)  # value of advection factor term
        self.d = CellVariable(d, mesh=self.mesh)  # value of diffusive factor
        self.s = CellVariable(s, mesh=self.mesh)  # Source term
        self.discretization = discretization
        self.theta = theta
        self.time_step = time_step

        # Check Peclet number

        mu = peclet_number(self.a, self.mesh.cell_widths, self.d)
        if np.max(np.abs(mu)) >= 1.5 and np.max(np.abs(mu)) < 2.0:
            warnings.warn(
                "\n\nThe Peclet number is %g, this is getting close to the limit of mod 2."
                % (np.max(np.abs(mu)),)
            )
        elif np.max(np.abs(mu)) > 2:
            warnings.warn(
                "\n\nThe Peclet number (%g) has exceeded the maximum value of mod 2 for the central discretization scheme."
                % (np.max(np.abs(mu)),)
            )

        # Check CFL condition
        CFL = CFL_condition(self.a, self.time_step, self.mesh.cell_widths)
        if np.max(np.abs(CFL)) > 0.5 and np.max(np.abs(CFL)) < 1.0:
            warnings.warn(
                "\n\nThe CFL condition value is %g, it is getting close to the upper limit."
                % (np.max(np.abs(CFL)),)
            )
        elif np.max(np.abs(CFL)) > 1:
            warnings.warn(
                "\n\nThe CFL condition value is %g, and has gone above the upper limit."
                % (np.max(np.abs(CFL)),)
            )

        if discretization == "exponential":
            self.kappa = (np.exp(mu) + 1) / (np.exp(mu) - 1) - 2 / mu
            self.kappa[np.where(mu == 0.0)] = 0
            self.kappa[np.where(np.isposinf(mu))] = 1
            self.kappa[np.where(np.isneginf(mu))] = -1
        elif discretization == "upwind":
            kappa_neg = np.where(self.a < 0, -1, 0)
            kappa_pos = np.where(self.a > 0, 1, 0)
            self.kappa = kappa_neg + kappa_pos
        elif discretization == "central":
            self.kappa = np.zeros(self.mesh.n_cells)
        else:
            print(
                "Please set `discretization` to one of the following: `upwind`, `central` or `exponential`."
            )

        # Artificially modify the diffusion coefficient to introduce adaptive discretization
        self.d = self.d + 0.5 * self.a * self.mesh.cell_widths * self.kappa
        print("Using kappa", np.min(self.kappa), np.max(self.kappa))
        print(self.kappa)

        # other variables that need to be initialized
        self.I = identity(self.mesh.n_cells)
        self.w = np.zeros(self.mesh.n_cells)

    def _interior_matrix_elements(self, i):
        # Interior coefficients for matrix equation
        ra = (
            lambda i, a, d, m: 1.0
            / m.h(i)
            * (a.m(i) * m.h(i) / (2 * m.hm(i)) + d.m(i) / m.hm(i))
        )
        rb = (
            lambda i, a, d, m: 1.0
            / m.h(i)
            * (
                a.m(i) * m.h(i - 1) / (2 * m.hm(i))
                - a.p(i) * m.h(i + 1) / (2 * m.hp(i))
                - d.m(i) / m.hm(i)
                - d.p(i) / m.hp(i)
            )
        )
        rc = (
            lambda i, a, d, m: 1.0
            / m.h(i)
            * (-a.p(i) * m.h(i) / (2 * m.hp(i)) + d.p(i) / m.hp(i))
        )
        return (
            ra(i, self.a, self.d, self.mesh),
            rb(i, self.a, self.d, self.mesh),
            rc(i, self.a, self.d, self.mesh),
        )

    def set_boundary_conditions(
        self,
        left_flux: float = None,  # type: ignore
        right_flux: float = None,  # type: ignore
        left_value: float = None,  # type: ignore
        right_value: float = None,  # type: ignore
    ):
        """Make sure this function is used sensibly otherwise the matrix will be ill posed."""

        self.left_flux = left_flux
        self.right_flux = right_flux
        self.left_value = left_value
        self.right_value = right_value

    def alpha_matrix(self):
        """The alpha matrix is used to mask boundary conditions values for Dirichlet
        conditions. Otherwise for a fully Neumann (or Robin) system it is equal to
        the identity matrix."""
        a1 = 0 if self.left_flux is None else 1
        aJ = 0 if self.right_flux is None else 1
        diagonals = np.ones(self.mesh.n_cells)
        diagonals[0] = a1
        diagonals[-1] = aJ
        return sparse.diags(diagonals, 0)

    def beta_vector(self):
        """Returns the robin boundary condition vector."""
        b = np.zeros(self.mesh.n_cells)

        if self.left_flux is not None:
            left_bc_elements = robin_boundary_condition_vector_elements_left(
                m=self.mesh, flux=self.left_flux
            )

        if self.right_flux is not None:
            right_bc_elements = robin_boundary_condition_vector_elements_right(
                m=self.mesh, flux=-self.right_flux
            )

        if self.left_value is not None:
            left_bc_elements = dirichlet_boundary_condition_vector_elements_left()

        if self.right_value is not None:
            right_bc_elements = dirichlet_boundary_condition_vector_elements_right()

        bcs = left_bc_elements + right_bc_elements  # type: ignore
        for inx, value in bcs:
            b[inx] = value
        return b

    def coefficient_matrix(self):
        """Returns the coefficient matrix which appears on the left hand side."""
        J = self.mesh.n_cells

        padding = np.array(
            [0]
        )  # A element which is pushed off the edge of the matrix by the spdiags function
        zero = padding  # Yes, its the same. But this element is included in the matrix (semantic difference).

        if self.left_flux is not None:
            left_bc_elements = robin_boundary_condition_matrix_elements_left(
                self.mesh, self.a, self.d
            )

        if self.right_flux is not None:
            right_bc_elements = robin_boundary_condition_matrix_elements_right(
                self.mesh, self.a, self.d
            )

        if self.left_value is not None:
            left_bc_elements = dirichlet_boundary_condition_matrix_elements_left()

        if self.right_value is not None:
            right_bc_elements = dirichlet_boundary_condition_matrix_elements_right()

        # Use the functions to layout the matrix Note that the boundary
        # condition elements are set to zero, they are filled in as
        # the next step.
        inx = np.array(range(1, J - 1))
        ra, rb, rc = self._interior_matrix_elements(inx)
        #                                 c1
        upper = np.concatenate([padding, zero, rc])

        #                          b1           bJ
        central = np.concatenate([zero, rb, zero])  # + self.s  # added source term here

        #                               aJ
        lower = np.concatenate([ra, zero, padding])

        A = sparse.spdiags([lower, central, upper], [-1, 0, 1], J, J).todok()

        # Apply boundary conditions elements
        bcs = left_bc_elements + right_bc_elements  # type: ignore
        for inx, value in bcs:
            A[inx] = value
        return dia_matrix(A)

    def initialize_system(self, w_init):
        self.M = self.coefficient_matrix()
        self.alpha = self.alpha_matrix()

        # Construct linear system from discretised matrices, A.x = d
        self.w = w_init
        self.A = self.I - self.time_step * self.theta * self.alpha * self.M

    def take_step(self, time_step=None):
        if time_step is None:
            time_step = self.time_step

        # we may need to update the other matrices (M and alpha) if BCs change with time.

        b = (
            self.I + time_step * (1 - self.theta) * self.alpha * self.M
        ) * self.w + self.beta_vector() * time_step
        self.w = spsolve(self.A, b)

        return self.w

    def solve_to_time(self, time_end):
        n_steps = int(time_end / self.time_step)
        for _ in tqdm(range(n_steps), desc="step"):
            self.take_step()

        return self.w
