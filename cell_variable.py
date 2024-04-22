import numpy as np

from .mesh import Mesh


class CellVariable(np.ndarray):
    """Representation of a variable defined at the cell centers. Provides interpolation functions to calculate the value at cell faces."""

    def __new__(cls, input_array, mesh: Mesh):
        # If `input_array` is actually just a constant
        # convert it to an array of len the number of cells.
        try:
            len(input_array)
        except TypeError:
            input_array = input_array * np.ones(len(mesh.cells))

        obj = np.asarray(input_array).view(cls)
        obj.mesh = mesh
        return obj

    def __array_finalize__(self, obj: Mesh):
        if obj is None:
            return
        self.mesh = getattr(obj, "mesh", None)
        self.__get_items__ = getattr(obj, "__get_items__", None)

    def m(self, i):
        """Linear interpolation of the cell value at the right hand face i.e. along the _m_inus direction."""
        return (
            self.mesh.h(i) / (2 * self.mesh.hm(i)) * self[i - 1]
            + self.mesh.h(i - 1) / (2 * self.mesh.hm(i)) * self[i]
        )

    def p(self, i):
        """Linear interpolation of the cell value at the right hand face i.e. along the _p_lus direction."""
        return (
            self.mesh.h(i + 1) / (2 * self.mesh.hp(i)) * self[i]
            + self.mesh.h(i) / (2 * self.mesh.hp(i)) * self[i + 1]
        )
