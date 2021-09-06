import numpy as np
import grid as g
import basis as b
import elliptic as ell
import plotter as my_plt
import fluxes as fx
import timestep as ts

# Parameters
order, res_x, res_y = 8, 25, 25
final_time, write_time = 1.5, 5.0e-2  # 1.5e0, 1.0e-2

# Flags
plot_ic = True
viscosity = True
nu = 2.0e-2  # value of viscosity

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
elsasser.initialize(grids=grids, numbers=[[2, 3, 4], [2, 3, 4]])
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

# Set-up fluxes and main loop
dg_flux = fx.DGFlux(resolutions=resolutions_ghosts, orders=orders,
                    viscosity=viscosity, nu=nu)
stepper = ts.Stepper(time_order=3, space_order=order,
                     write_time=write_time, final_time=final_time)

print('\nBeginning main loop...')
stepper.main_loop(elsasser=elsasser, basis=basis,
                  elliptic=elliptic, grids=grids, dg_flux=dg_flux)

print('\nVisualizing stop time condition')
if plot_ic:
    elsasser.convert_to_basic_variables()
    # check it out
    # plotter.vector_contourf(vector=elsasser.plus, titles=['Px', 'Py'])
    # plotter.vector_contourf(vector=elsasser.minus, titles=['Mx', 'My'])
    # plotter.vector_contourf(vector=elsasser.velocity, titles=['vx', 'vy'])
    # plotter.vector_contourf(vector=elsasser.magnetic, titles=['Bx', 'By'])
    plotter.vorticity_contourf(vector_arr=elsasser.velocity.arr, grids=grids, title='fluid vorticity')
    plotter.vorticity_contourf(vector_arr=elsasser.magnetic.arr, grids=grids, title='current density')
    plotter.scalar_contourf(scalar=elliptic.pressure)

    # make movie
    plotter.animate2d(stepper=stepper, grids=grids)

    # show all
    plotter.show()
