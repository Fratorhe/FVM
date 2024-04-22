def peclet_number(a, cell_widths, d):
    """
    Calculate the Peclet number for a given system.

    Parameters
    ----------
    a : float
        Advection velocity.
    cell_widths : array_like
        Array of cell widths.
    d : float
        Diffusion coefficient.

    Returns
    -------
    ndarray
        Array of Peclet numbers for each cell.
    """
    return a * cell_widths / d


def CFL_condition(a, k, cell_widths):
    """
    Calculate the CFL condition for a given system.

    Parameters
    ----------
    a : float
        Advection velocity.
    k : float
        Time step factor.
    cell_widths : array_like
        Array of cell widths.

    Returns
    -------
    ndarray
        Array of CFL conditions for each cell.
    """
    return a * k / cell_widths
