import numpy as np
import cupy as cp


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr, axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, orders, viscosity=False):
        # flags
        self.viscosity = viscosity

        # parameters
        self.resolutions = resolutions
        self.orders = orders

        # book-keeping: permutations
        self.permutations = [(0, 1, 4, 2, 3),
                             (0, 1, 2, 3, 4)]

        # more book-keeping: axis-wise boundary slices
        self.boundary_slices = [
            # x-directed face slices [(comps), (left), (right)]
            [(slice(2), slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1])),
             (slice(2), slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]))],
            # y-directed face slices [(left), (right)]
            [(slice(2), slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0),
             (slice(2), slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1)]]
        # Flux speed slices [(comps), (left), (right)]
        self.speed_slices = [
            # x-directed face slices [(left), (right)]
            [(slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1])),
             (slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]))],
            # y-directed face slices [(left), (right)]
            [(slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0),
             (slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1)]]

        # Grid and sub-element axes
        self.grid_axis = np.array([1, 3])
        self.sub_element_axis = np.array([2, 4])
        # Numerical flux allocation size arrays
        self.num_flux_sizes = [(2, resolutions[0], 2, resolutions[1], orders[1]),
                               (2, resolutions[0], orders[0], resolutions[1], 2)]

    def semi_discrete_rhs(self, elsasser, elliptic, basis, grids):
        """
        Calculate the right-hand side of semi-discrete equation
        """
        return (((self.x_flux(vector=elsasser.plus, advector=elsasser.minus,
                              basis=basis.basis_x) * grids.x.J) +
                (self.y_flux(vector=elsasser.plus, advector=elsasser.minus,
                             basis=basis.basis_y) * grids.y.J) +
                self.source_term(elliptic=elliptic, vector=elsasser.plus, grids=grids)),
                ((self.x_flux(vector=elsasser.minus, advector=elsasser.plus,
                              basis=basis.basis_x) * grids.x.J) +
                 (self.y_flux(vector=elsasser.minus, advector=elsasser.plus,
                              basis=basis.basis_y) * grids.y.J) +
                 self.source_term(elliptic=elliptic, vector=elsasser.minus, grids=grids)))

    def x_flux(self, vector, advector, basis):  # , elliptic, grid_x):
        dim = 0
        # Advection: flux is the tensor v_i * v_j
        flux = advector.arr[0, :, :, :, :] * vector.arr[:, :, :, :, :]
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.spatial_flux(flux=flux, speed=vector.arr, basis=basis, dim=dim))

    def y_flux(self, vector, advector, basis):  # , elliptic, grid_y):
        dim = 1
        # Advection: flux is the tensor v_i * v_j
        flux = advector.arr[1, :, :, :, :] * vector.arr[:, :, :, :, :]
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.spatial_flux(flux=flux, speed=vector.arr, basis=basis, dim=dim))

    def spatial_flux(self, flux, speed, basis, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])

        # Measure upwind directions
        speed_neg = cp.where(condition=speed[dim, :, :, :, :] < 0, x=1, y=0)
        speed_pos = cp.where(condition=speed[dim, :, :, :, :] >= 0, x=1, y=0)

        # Upwind flux, left and right faces
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1, axis=self.grid_axis[dim]),
                                                                     speed_pos[self.speed_slices[dim][0]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     speed_neg[self.speed_slices[dim][0]]))
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              speed_pos[self.speed_slices[dim][1]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              speed_neg[self.speed_slices[dim][1]]))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])

    def source_term(self, elliptic, vector, grids):
        """
        Add source term in ideal MHD momentum equation point-wise: the pressure gradient
        future work: experimental_viscosity
        """
        nu = 1.0e-2

        if self.viscosity:
            return nu * vector.laplacian(grids=grids) - elliptic.pressure_gradient.arr
        else:
            return -1.0 * elliptic.pressure_gradient.arr
