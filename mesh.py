from collections.abc import Iterable

import numpy as np


def check_index_within_bounds(i, min_i, max_i):
    """Checks that the index specified (can be number or an iterable) is within the given range."""
    success = np.all((i >= min_i) * (i <= max_i))
    if success:
        return True

    if isinstance(i, Iterable):
        # The index is array-like
        idx = i[np.where(np.logical_not(np.logical_and(i >= min_i, i <= max_i)))]
        print(f"Index is out of bounds.\ni={idx}")
    else:
        # The index is a number
        print(f"Index is out of bounds.\ni={i}")
    return False


class Mesh:
    """A 1D cell centered mesh defined by faces for the finite volume method."""

    def __init__(self, faces):

        # Check for duplicated points
        if len(faces) != len(set(faces)):
            raise ValueError(
                "The faces array contains duplicated positions. No cell can have zero volume so please update with unique face positions."
            )
        self.faces = np.array(faces)  # faces locations
        self.cells = 0.5 * (self.faces[0:-1] + self.faces[1:])  # centers of the cells
        self.n_cells = len(self.cells)  # number of cells
        self.cell_widths = self.faces[1:] - self.faces[0:-1]  # width of the cells

    def h(self, i):
        """Returns the width of the cell at the specified index."""
        return self.cell_widths[i]

    def hm(self, i):
        """Distance between cell centers in the backwards direction."""
        if not check_index_within_bounds(i, 1, self.n_cells - 1):
            raise ValueError("hm index runs out of bounds")
        return self.cells[i] - self.cells[i - 1]

    def hp(self, i):
        """Distance between cell centers in the forward direction."""
        if not check_index_within_bounds(i, 0, self.n_cells - 2):
            raise ValueError("hp index runs out of bounds")
        return self.cells[i + 1] - self.cells[i]

    def move_mesh(self):
        self.faces = self.faces

    def dump_mesh(self, filename):
        pass

    ##### class methods to instantiate the Mesh given different options #####

    @classmethod
    def uniform(cls, a, b, n_points=50):
        return cls(np.linspace(a, b, n_points))

    @classmethod
    def simple_grading(cls, a, b, n_points=50, ratio=1):
        """
        Class method to initialize using a simple grading from `a` to `b`, similar to OpenFOAM.


        Parameters
        ----------
        a : float
            The starting point of the grading.
        b : float
            The end point of the grading.
        n_points : int, optional
            The number of points in the grading, defaults to 50.
        ratio : float, optional
            The growth ratio between consecutive points, defaults to 1.
            If ratio > 1, more points near b.

        Returns
        -------
        cls
            A new instance of the class with the specified grading.
        """

        length_slab = b - a
        growth_rate = ratio ** (1 / (n_points - 2))
        ds = length_slab / np.sum(growth_rate ** np.arange(n_points - 1))
        de = ratio * ds  # mm - last element height in mm
        dl = de / growth_rate ** (
            np.arange(n_points - 1)
        )  # length of specified element in mm
        cumulative_dl = np.cumsum(np.insert(dl, 0, 0))
        return cls(cumulative_dl)
