import numpy as np
import grid as g
import basis as b
import elliptic as ell
import plotter as my_plt

# Parameters
order, res_x, res_y = 8, 25, 25

# Flags
plot_ic = True

print('Setting up basis, grids, and variables')

# Build basis
orders = np.array([order, order])
basis = b.Basis2D(orders)

# Initialize grids
L = 2.0 * np.pi
lows = np.array([-L / 2.0, -L / 2.0])
highs = np.array([L / 2.0, L / 2.0])
resolutions = np.array([res_x, res_y])
resolutions_ghosts = np.array([res_x + 2, res_y + 2])
grids = g.Grid2D(basis=basis, lows=lows, highs=highs, resolutions=resolutions, linspace=True)

# Initialize variables
elsasser = g.Elsasser(resolutions=resolutions_ghosts, orders=orders)
elsasser.initialize(grids=grids)
elsasser.convert_to_basic_variables()

# Initialize elliptic class and pressure solve
elliptic = ell.Elliptic(grids=grids)
elliptic.pressure_solve(elsasser=elsasser, grids=grids)

# Set-up plotter class
plotter = my_plt.Plotter2D(grids=grids)

print('\nVisualizing initial condition')
if plot_ic:
    # plotter.vector_contourf(vector=elsasser.plus, titles=['Px', 'Py'])
    # plotter.vector_contourf(vector=elsasser.minus, titles=['Mx', 'My'])
    # plotter.vector_contourf(vector=elsasser.velocity, titles=['vx', 'vy'])
    # plotter.vector_contourf(vector=elsasser.magnetic, titles=['Bx', 'By'])
    plotter.scalar_contourf(scalar=elliptic.pressure)
    plotter.show()
