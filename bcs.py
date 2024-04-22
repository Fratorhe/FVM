from functools import partial

# ROBIN BOUNDARY CONDITONS


def robin_boundary_condition_matrix_elements_left(m, a, d):
    # Left side Robin boundary coefficients for matrix equation
    b1 = 1.0 / m.h(0) * (-a.p(0) * m.h(1) / (2 * m.hp(0)) - d.p(0) / m.hp(0))
    c1 = 1.0 / m.h(0) * (-a.p(0) * m.h(0) / (2 * m.hp(0)) + d.p(0) / m.hp(0))

    # Index and element value
    locations = [(0, 0), (0, 1)]
    values = (b1, c1)
    return tuple([list(x) for x in zip(locations, values)])


def robin_boundary_condition_matrix_elements_right(m, a, d):
    # Right hand side Robin boundary coefficients for matrix equation
    J = m.n_cells
    aJ = (
        1.0
        / m.h(J - 1)
        * (a.m(J - 1) * m.h(J - 1) / (2 * m.hm(J - 1)) + d.m(J - 1) / m.hm(J - 1))
    )
    bJ = (
        1.0
        / m.h(J - 1)
        * (a.m(J - 1) * m.h(J - 2) / (2 * m.hm(J - 1)) - d.m(J - 1) / m.hm(J - 1))
    )

    J = m.n_cells  # Index and element value

    # Index and element value
    locations = [(J - 1, J - 2), (J - 1, J - 1)]
    values = (aJ, bJ)
    return tuple([list(x) for x in zip(locations, values)])


def robin_boundary_condition_vector_elements(location, m, flux):
    location = [location]
    value = [flux / m.h(location)]
    return tuple([list(x) for x in zip(location, value)])


robin_boundary_condition_vector_elements_left = partial(
    robin_boundary_condition_vector_elements, location=0
)

robin_boundary_condition_vector_elements_right = partial(
    robin_boundary_condition_vector_elements, location=-1
)

# DIRICHLET BOUNDARY CONDITIONS


def dirichlet_boundary_conditions(locations, values):
    # Dirichlet boundary coefficients
    return tuple([list(x) for x in zip(locations, values)])


# Left hand side Dirichlet boundary coefficients for matrix equation
dirichlet_boundary_condition_matrix_elements_left = partial(
    dirichlet_boundary_conditions, locations=[(0, 0), (0, 1)], values=(0, 1)
)

dirichlet_boundary_condition_matrix_elements_right = partial(
    dirichlet_boundary_conditions, locations=[(-1, -2), (-1, -1)], values=(0, 1)
)

dirichlet_boundary_condition_vector_elements_left = partial(
    dirichlet_boundary_conditions, locations=(0,), values=(0,)
)

dirichlet_boundary_condition_vector_elements_right = partial(
    dirichlet_boundary_conditions, locations=(-1,), values=(0,)
)
