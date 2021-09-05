import numpy as np
import cupy as cp
import time as timer
import grid as g


# Dictionaries
ssp_rk_switch = {
    1: [1],
    2: [1 / 2, 1 / 2],
    3: [1 / 3, 1 / 2, 1 / 6],
    4: [3 / 8, 1 / 3, 1 / 4, 1 / 24],
    5: [11 / 30, 3 / 8, 1 / 6, 1 / 12, 1 / 120],
    6: [53 / 144, 11 / 30, 3 / 16, 1 / 18, 1 / 48, 1 / 720],
    7: [103 / 280, 53 / 144, 11 / 60, 3 / 48, 1 / 72, 1 / 240, 1 / 5040],
    8: [2119 / 5760, 103 / 280, 53 / 288, 11 / 180, 1 / 64, 1 / 360, 1 / 1440, 1 / 40320]
}

# Courant numbers for RK-DG stability from Cockburn and Shu 2001, [time_order][space_order-1]
courant_numbers = {
    1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    2: [1.0, 0.333],
    3: [1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033],
    4: [1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037],
    5: [1.608, 0.534, 0.271, 0.167, 0.115, 0.085, 0.065, 0.052, 0.042],
    6: [1.776, 0.592, 0.300, 0.185, 0.127, 0.093, 0.072, 0.057, 0.047],
    7: [1.977, 0.659, 0.333, 0.206, 0.142, 0.104, 0.080, 0.064, 0.052],
    8: [2.156, 0.718, 0.364, 0.225, 0.154, 0.114, 0.087, 0.070, 0.057]
}

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, time_order, space_order, write_time, final_time, linear=False):
        # Time-stepper order and SSP-RK coefficients
        self.time_order = time_order
        self.space_order = space_order
        if linear:
            self.coefficients = self.get_coefficients()
        else:
            self.coefficients = self.get_nonlinear_coefficients()

        # Courant number
        self.courant = self.get_courant_number()

        # Simulation time init
        self.time = 0
        self.dt = None
        self.steps_counter = 0
        self.write_counter = 1  # IC already written

        # Time between write-outs
        self.write_time = write_time
        # Final time to step to
        self.final_time = final_time

        # Field energy and time array
        self.time_array = np.array([self.time])
        self.field_energy = np.array([])

        # Stored array
        self.saved_times = []
        self.saved_array = []

    def get_coefficients(self):
        return np.array([ssp_rk_switch.get(self.time_order, "nothing")][0])

    def get_nonlinear_coefficients(self):
        return np.array(nonlinear_ssp_rk_switch.get(self.time_order, "nothing"))

    def get_courant_number(self):
        return courant_numbers.get(self.time_order)[self.space_order - 1]

    def main_loop(self, elsasser, basis, elliptic, grids, dg_flux):
        """
        Main loop for RK time-stepping algorithm
        """
        t0 = timer.time()
        self.adapt_time_step(max_speeds=get_max_speeds(elsasser=elsasser),
                             pressure_dt=estimate_pressure_dt(elsasser=elsasser, elliptic=elliptic),
                             dx=grids.x.dx, dy=grids.y.dx)
        # self.saved_array += [elsasser.plus.arr.get(), elsasser.minus.arr.get()]
        # self.saved_times += [self.time]

        while self.time < self.final_time:
            self.nonlinear_ssp_rk(elsasser=elsasser, basis=basis, elliptic=elliptic,
                                  grids=grids, dg_flux=dg_flux)
            # update time and steps
            self.time += self.dt.get()
            self.steps_counter += 1
            # Get time
            # self.time_array = np.append(self.time_array, self.time)
            # Do write-out sometimes
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                self.write_counter += 1
                # Filter
                elsasser.plus.filter(grids=grids)
                elsasser.minus.filter(grids=grids)
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt.get()))
                print('Time since start is {:0.3f}'.format((timer.time() - t0) / 60.0) + ' minutes')
            if cp.isnan(elsasser.plus.arr).any() or cp.isnan(elsasser.minus.arr).any():
                print('\nCaught a nan, exiting simulation.')
                return
            print('\nFinal time reached, finishing simulation')
            print('Total steps were ' + str(self.steps_counter))

    def nonlinear_ssp_rk(self, elsasser, basis, elliptic, grids, dg_flux):
        elsasser.ghost_sync()
        # Set up stages
        stage0 = g.Elsasser(resolutions=grids.res_ghosts, orders=grids.orders)
        stage1 = g.Elsasser(resolutions=grids.res_ghosts, orders=grids.orders)
        stage2 = g.Elsasser(resolutions=grids.res_ghosts, orders=grids.orders)
        stage0.plus.arr, stage0.minus.arr = cp.zeros_like(elsasser.plus.arr), cp.zeros_like(elsasser.plus.arr)
        stage1.plus.arr, stage1.minus.arr = cp.zeros_like(elsasser.plus.arr), cp.zeros_like(elsasser.plus.arr)
        stage2.plus.arr, stage2.minus.arr = cp.zeros_like(elsasser.plus.arr), cp.zeros_like(elsasser.plus.arr)

        # zeroth stage
        elliptic.pressure_solve(elsasser=elsasser, grids=grids)
        self.adapt_time_step(max_speeds=get_max_speeds(elsasser=elsasser),
                             pressure_dt=estimate_pressure_dt(elsasser=elsasser, elliptic=elliptic),
                             dx=grids.x.dx, dy=grids.y.dx)
        # to-do: complete shu-osher ssprk time-stepping routine

    def adapt_time_step(self, max_speeds, pressure_dt, dx, dy):
        max0_wp = max_speeds[0]  # + np.sqrt(max_pressure)
        max1_wp = max_speeds[1]  # + np.sqrt(max_pressure)
        self.dt = self.courant / ((max0_wp / dx) + (max1_wp / dy) +
                                  1.0 / pressure_dt[0] + 1.0 / pressure_dt[1]) / 4.0 / 2.0
        # vis_dt = self.courant * dx * dx / 1.0e0 / (2.0 ** 0.5)


def get_max_speeds(elsasser):
    return cp.array([cp.amax(cp.absolute(elsasser.plus.arr[0, :, :, :, :])),
                     cp.amax(cp.absolute(elsasser.plus.arr[1, :, :, :, :])),
                     cp.amax(cp.absolute(elsasser.minus.arr[0, :, :, :, :])),
                     cp.amax(cp.absolute(elsasser.minus.arr[1, :, :, :, :]))
                     ])


def estimate_pressure_dt(elsasser, elliptic):
    return cp.array([cp.amax(cp.absolute(elsasser.plus.arr[0, :, :, :, :])) /
                     cp.amax(cp.absolute(elliptic.pressure_gradient.arr[0, :, :, :, :])),
                     cp.amax(cp.absolute(elsasser.plus.arr[1, :, :, :, :])) /
                     cp.amax(cp.absolute(elliptic.pressure_gradient.arr[1, :, :, :, :])),
                     cp.amax(cp.absolute(elsasser.minus.arr[0, :, :, :, :])) /
                     cp.amax(cp.absolute(elliptic.pressure_gradient.arr[0, :, :, :, :])),
                     cp.amax(cp.absolute(elsasser.minus.arr[1, :, :, :, :])) /
                     cp.amax(cp.absolute(elliptic.pressure_gradient.arr[1, :, :, :, :]))
                     ])
