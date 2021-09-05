import numpy as np
import cupy as cp
import grid as g


class Elliptic:
    def __init__(self, grids):
        # Resolutions
        self.orders = np.array([grids.x.order, grids.y.order])
        self.resolution_ghosts = np.array([grids.x.res_ghosts, grids.y.res_ghosts])

        # Pressure scalar object
        self.pressure = g.Scalar(resolutions=self.resolution_ghosts, orders=self.orders)
        self.pressure.arr = cp.zeros((self.resolution_ghosts[0], self.orders[0],
                                      self.resolution_ghosts[1], self.orders[1]))

        # Pressure gradient vector object
        self.pressure_gradient = g.Vector(resolutions=self.resolution_ghosts, orders=self.orders)
        self.pressure_gradient.arr = cp.zeros((2,
                                               self.resolution_ghosts[0], self.orders[0],
                                               self.resolution_ghosts[1], self.orders[1]))

        # Spectral indicators
        self.kr_sq = (outer2(grids.x.d_wave_numbers, cp.ones_like(grids.y.d_wave_numbers)) ** 2.0 +
                      outer2(cp.ones_like(grids.x.d_wave_numbers), grids.y.d_wave_numbers) ** 2.0)

    def pressure_solve(self, elsasser, grids):
        """
        Solve Poisson equation for pressure
        """
        # Compute velocity-gradient tensor and its contraction
        elsasser.poisson_source(grids=grids)

        # Fourier transform pressure source
        spectrum = grids.fourier_transform(function=elsasser.pressure_source)

        # Determine Poisson solution spectrum (nan_to_num takes divide-by-zero nan to zero)
        poisson_spectrum = -1.0 * cp.nan_to_num(cp.divide(spectrum, self.kr_sq))  # self.ikx ** 2.0 + self.iky ** 2.0))

        # Inverse transform for poisson solution
        self.pressure.arr[1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=poisson_spectrum)

        # Compute components of pressure gradient
        x_spectrum = cp.multiply(1j * grids.x.d_wave_numbers[:, None], poisson_spectrum)
        y_spectrum = cp.multiply(1j * grids.y.d_wave_numbers[None, :], poisson_spectrum)

        self.pressure_gradient.arr[0, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=x_spectrum)
        self.pressure_gradient.arr[1, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=y_spectrum)


def outer2(a, b):
    """
    Compute outer tensor product of vectors a, b
    :param a: vector a_i
    :param b: vector b_j
    :return: tensor a_i b_j
    """
    return cp.tensordot(a, b, axes=0)