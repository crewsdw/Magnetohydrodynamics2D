import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation


class Plotter2D:
    """
    Creates plots of functions on 2D piecewise grid using matplotlib
    """

    def __init__(self, grids, colormap='RdPu'):
        self.colormap = colormap
        order = grids.orders[0]
        # Build structured grid
        self.X, self.Y = np.meshgrid(grids.x.arr[:, :].flatten(),
                                     grids.y.arr[:, :].flatten(), indexing='ij')
        # no ghost slices
        self.ng = (slice(order, -order), slice(order, -order))
        self.ng0 = (0, slice(order, -order), slice(order, -order))
        self.ng1 = (1, slice(order, -order), slice(order, -order))
        self.nt00 = (0, 0, slice(None), slice(None))
        self.nt01 = (0, 1, slice(None), slice(None))
        self.nt10 = (1, 0, slice(None), slice(None))
        self.nt11 = (1, 1, slice(None), slice(None))

        # Build linearly spaced fourier-interpolated grid
        self.XE = np.tensordot(grids.x.arr_lin, np.ones_like(grids.y.arr_lin), axes=0)
        self.YE = np.tensordot(np.ones_like(grids.x.arr_lin), grids.y.arr_lin, axes=0)

        # For streamplots, starting points
        self.L = grids.x.length
        self.start_points = np.array([np.random.random_sample(300) * self.L - self.L / 2,
                                      np.random.random_sample(300) * self.L - self.L / 2]).transpose()

    def vector_contourf(self, vector, titles=None):
        if titles is None:
            titles = ["vector x-component", 'vector y-component']
        cb = np.linspace(cp.amin(vector.arr), cp.amax(vector.arr), num=100).get()

        plt.figure()
        plt.contourf(self.X[self.ng], self.Y[self.ng],
                     vector.grid_flatten_arr().get()[self.ng0],
                     cb, cmap=self.colormap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(titles[0])
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        plt.contourf(self.X[self.ng], self.Y[self.ng],
                     vector.grid_flatten_arr().get()[self.ng1],
                     cb, cmap=self.colormap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(titles[1])
        plt.colorbar()
        plt.tight_layout()

    def scalar_contourf(self, scalar, title='pressure'):
        cb = np.linspace(cp.amin(scalar.arr), cp.amax(scalar.arr), num=100).get()

        plt.figure()
        plt.contourf(self.X[self.ng], self.Y[self.ng],
                     scalar.grid_flatten_arr().get()[self.ng],
                     cb, cmap=self.colormap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

    def divergence_contourf(self, vector_arr, grids, title='divergence'):
        """
        Plots divergence of vector array value (not Vector object)
        """
        # Compute vector divergence
        spectrum_x = grids.fourier_transform(function=cp.asarray(
                                            vector_arr[0, 1:-1, :, 1:-1, :]))
        spectrum_y = grids.fourier_transform(function=cp.asarray(
                                            vector_arr[1, 1:-1, :, 1:-1, :]))
        divergence = grids.inverse_transform_linspace(
            spectrum=(cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_x) +
                      cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_y))).get()

        # Plot divergence contours
        cb = np.linspace(np.amin(divergence), np.amax(divergence), num=100)
        plt.figure()
        plt.contourf(self.XE, self.YE, divergence,
                     cb, cmap=self.colormap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

    def vorticity_contourf(self, vector_arr, grids, title='vorticity'):
        """
        Plots vorticity of vector array value (not Vector object)
        """
        # Compute vector vorticity
        spectrum_x = grids.fourier_transform(function=cp.asarray(
            vector_arr[0, 1:-1, :, 1:-1, :]))
        spectrum_y = grids.fourier_transform(function=cp.asarray(
            vector_arr[1, 1:-1, :, 1:-1, :]))
        vorticity = grids.inverse_transform_linspace(
            spectrum=(cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_y) -
                      cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_x))).get()

        # Plot vorticity contours
        cb = np.linspace(np.amin(vorticity), np.amax(vorticity), num=100)
        plt.figure()
        plt.contourf(self.XE, self.YE, vorticity,
                     cb, cmap=self.colormap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

    def streamlines2d(self, vector, grids, title='instantaneous streamlines'):
        # fourier interpolation
        spectrum_x = grids.fourier_transform(function=vector.arr[0, 1:-1, :, 1:-1, :])
        spectrum_y = grids.fourier_transform(function=vector.arr[1, 1:-1, :, 1:-1, :])
        UE = grids.inverse_transform_linspace(spectrum=spectrum_x)
        VE = grids.inverse_transform_linspace(spectrum=spectrum_y)
        V = np.sqrt(UE.get() ** 2.0 + VE.get() ** 2.0).transpose()

        plt.figure()
        plt.streamplot(self.YE, self.XE,
                       UE.get().transpose(), VE.get().transpose(),
                       density=5.0, start_points=self.start_points, color=V)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.tight_layout()

    def show(self):
        plt.show()