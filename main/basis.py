import numpy as np
import cupy as cp
import scipy.special as sp

# Legendre-Gauss-Lobatto nodes and quadrature weights dictionaries
lgl_nodes = {
    1: [0],
    2: [-1, 1],
    3: [-1, 0, 1],
    4: [-1, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1],
    5: [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1],
    6: [-1, -np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), -np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
        np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21), np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), 1],
    7: [-1, -0.830223896278566929872, -0.468848793470714213803772,
        0, 0.468848793470714213803772, 0.830223896278566929872, 1],
    8: [-1, -0.8717401485096066153375, -0.5917001814331423021445,
        -0.2092992179024788687687, 0.2092992179024788687687,
        0.5917001814331423021445, 0.8717401485096066153375, 1],
    9: [-1, -0.8997579954114601573124, -0.6771862795107377534459,
        -0.3631174638261781587108, 0, 0.3631174638261781587108,
        0.6771862795107377534459, 0.8997579954114601573124, 1],
    10: [-1, -0.9195339081664588138289, -0.7387738651055050750031,
         -0.4779249498104444956612, -0.1652789576663870246262,
         0.1652789576663870246262, 0.4779249498104444956612,
         0.7387738651055050750031, 0.9195339081664588138289, 1]
}

lgl_weights = {
    1: [2],
    2: [1, 1],
    3: [1 / 3, 4 / 3, 1 / 3],
    4: [1 / 6, 5 / 6, 5 / 6, 1 / 6],
    5: [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10],
    6: [1 / 15, (14 - np.sqrt(7)) / 30, (14 + np.sqrt(7)) / 30,
        (14 + np.sqrt(7)) / 30, (14 - np.sqrt(7)) / 30, 1 / 15],
    7: [0.04761904761904761904762, 0.2768260473615659480107,
        0.4317453812098626234179, 0.487619047619047619048,
        0.4317453812098626234179, 0.2768260473615659480107,
        0.04761904761904761904762],
    8: [0.03571428571428571428571, 0.210704227143506039383,
        0.3411226924835043647642, 0.4124587946587038815671,
        0.4124587946587038815671, 0.3411226924835043647642,
        0.210704227143506039383, 0.03571428571428571428571],
    9: [0.02777777777777777777778, 0.1654953615608055250463,
        0.2745387125001617352807, 0.3464285109730463451151,
        0.3715192743764172335601, 0.3464285109730463451151,
        0.2745387125001617352807, 0.1654953615608055250463,
        0.02777777777777777777778],
    10: [0.02222222222222222222222, 0.1333059908510701111262,
         0.2248893420631264521195, 0.2920426836796837578756,
         0.3275397611838974566565, 0.3275397611838974566565,
         0.292042683679683757876, 0.224889342063126452119,
         0.133305990851070111126, 0.02222222222222222222222]
}


# lagrange interpolation functions
# inspired by stackexchange: https://stackoverflow.com/questions/4003794/lagrange-interpolation-in-python
# class LagrangePoly:
#     def __init__(self, X, Y):
#         self.n = len(X)
#         self.X = np.array(X)
#         self.Y = np.array(Y)
#
#     def basis(self, x, j):
#         b = [(x - self.X[m]) / (self.X[j] - self.X[m])
#              for m in range(self.n) if m != j]
#         return np.prod(b, axis=0) * self.Y[j]
#
#     def deriv(self, i, j):  # derivative of i'th basis at node j
#         if (i == j):
#             d = [1 / (self.X[i] - self.X[m]) for m in range(self.n) if m != i]
#             return np.sum(d, axis=0)
#         if (i != j):
#             num = [(self.X[j] - self.X[m]) for m in range(self.n) if m != i and m != j]
#             den = [(self.X[i] - self.X[m]) for m in range(self.n) if m != i]
#             return (np.prod(num, axis=0) / np.prod(den, axis=0))
#
#     def interpolate(self, x):
#         b = [self.basis(x, j) for j in range(self.n)]
#         return np.sum(b, axis=0)


class Basis1D:
    def __init__(self, order):
        self.order = int(order)
        self.nodes = self.get_nodes()
        self.weights = self.get_weights()
        self.eigenvalues = self.set_eigenvalues()

        # Vandermonde and inverse
        self.vandermonde = self.set_vandermonde()
        self.vandermonde_inverse = self.set_vandermonde_inverse()

        # Mass matrix and inverse
        self.mass = self.mass_matrix()
        self.d_mass = cp.asarray(self.mass)
        self.invm = self.inv_mass_matrix()
        self.face_mass = np.eye(self.order)[:, np.array([0, -1])]  # face mass, first and last columns of identity

        # Inner product arrays
        self.adv = self.advection_matrix()
        self.stf = self.adv.T

        # DG weak form arrays, numerical flux is first and last columns of inverse mass matrix
        # both are cupy arrays
        self.up = self.internal_flux()
        self.xi = cp.asarray(self.invm[:, np.array([0, -1])])
        # numpy array form
        self.np_up = self.up.get()
        self.np_xi = self.xi.get()

        # DG strong form array
        self.der = self.derivative_matrix()

    def get_nodes(self):
        nodes = lgl_nodes.get(self.order, "nothing")
        return nodes

    def get_weights(self):
        weights = lgl_weights.get(self.order, "nothing")
        return weights

    def set_eigenvalues(self):
        # Legendre-Lobatto "eigenvalues"
        eigenvalues = np.array([(2.0 * s + 1) / 2.0 for s in range(self.order - 1)])

        # if self.order == 1:
        #    eigenvalues = 1 / 2
        return np.append(eigenvalues, (self.order - 1) / 2.0)

    def set_vandermonde(self):
        return np.array([[sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_vandermonde_inverse(self):
        return np.array([[self.weights[j] * self.eigenvalues[s] * sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def mass_matrix(self):
        # Diagonal part
        approx_mass = np.diag(self.weights)

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = np.multiply(self.weights, p(self.nodes))
        a = -self.order * (self.order - 1) / (2 * (2 * self.order - 1))
        # calculate mass matrix
        return approx_mass + a * np.outer(v, v)

    def advection_matrix(self):
        adv = np.zeros((self.order, self.order))

        # Fill matrix
        for i in range(self.order):
            for j in range(self.order):
                adv[i, j] = self.weights[i] * self.weights[j] * sum(
                    self.eigenvalues[s] * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clean machine error
        adv[np.abs(adv) < 1.0e-15] = 0

        return adv

    def inv_mass_matrix(self):
        # Diagonal part
        approx_inv = np.diag(np.divide(1.0, self.weights))

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = p(self.nodes)
        b = self.order / 2
        # calculate inverse mass matrix
        return approx_inv + b * np.outer(v, v)

    def internal_flux(self):
        # Compute internal flux array
        up = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                up[i, j] = self.weights[j] * sum(
                    (2 * s + 1) / 2 * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clear machine errors
        up[np.abs(up) < 1.0e-10] = 0

        return cp.asarray(up)

    def derivative_matrix(self):
        der = np.zeros((self.order, self.order))

        for i in range(self.order):
            for j in range(self.order):
                der[i, j] = self.weights[j] * sum(
                    self.eigenvalues[s] * sp.legendre(s).deriv()(self.nodes[i]) *
                    sp.legendre(s)(self.nodes[j]) for s in range(self.order))

        # Clear machine errors
        der[np.abs(der) < 1.0e-15] = 0

        return der

    def fourier_transform_array(self, midpoints, J, wave_numbers):
        """
        Grid-dependent spectral coefficient matrix
        Needs grid quantities: Jacobian, wave numbers, nyquist number
        """
        # Check sign of wave-numbers (see below)
        signs = np.sign(wave_numbers)
        signs[np.where(wave_numbers == 0)] = 1.0

        # Fourier-transformed modal basis ( (-1)^s accounts for scipy's failure to have negative spherical j argument )
        p_tilde = np.array([(signs ** s) * np.exp(-1j * np.pi / 2.0 * s) *
                            sp.spherical_jn(s, np.absolute(wave_numbers) / J) for s in range(self.order)])

        # Multiply by inverse Vandermonde transpose for fourier-transformed nodal basis
        ell_tilde = np.matmul(self.vandermonde_inverse.T, p_tilde)

        # Outer product with phase factors
        phase = np.exp(-1j * np.tensordot(midpoints, wave_numbers, axes=0))
        transform_array = np.multiply(phase[:, :, None], ell_tilde.T)

        # Put in order (resolution, nodes, modes)
        transform_array = np.transpose(transform_array, (0, 2, 1))

        # Return as cupy array
        return cp.asarray(transform_array)

    def inverse_transform_array(self, midpoints, J, wave_numbers):
        """
        Grid-dependent spectral coefficient matrix
        Experimental inverse-transform matrix
        """
        # Check sign (see below)
        signs = np.sign(wave_numbers)
        signs[np.where(wave_numbers == 0)] = 1.0
        # wave_numbers[np.where(wave_numbers == 0)] = 1.0

        # Inverse transform array
        # p_tilde = np.array([(signs ** s) * np.exp(1j * np.pi / 2.0 * s) *
        #                     (sp.spherical_jn(s, np.absolute(wave_numbers) / J)) for s in range(self.order)])
        # Multiply by Vandermonde matrix
        # next_step = np.matmul(self.vandermonde.T, p_tilde)
        vandermonde_contraction = np.array(sum(self.vandermonde[i, :] for i in range(self.order)))
        spherical_summation = np.array(sum((signs ** s) * ((-1j) ** s) *  # np.exp(-1j * np.pi / 2.0 * s) *
                                (sp.spherical_jn(s, np.absolute(wave_numbers) / J)) for s in range(self.order)))
        # print(vandermonde_contraction.shape)
        # print(spherical_contraction.shape)

        # Outer product with phase factors
        phase = np.exp(1j * np.tensordot(midpoints, wave_numbers, axes=0))
        next_step = np.divide(phase, spherical_summation[None, :])
        inverse_transform = np.transpose(np.tensordot(next_step, vandermonde_contraction, axes=0),
                                         axes=(0, 2, 1))
        # print(inverse_transform.shape)
        # inverse_transform = np.multiply(phase[:, None, :], next_step[None, :, :])

        return cp.asarray(inverse_transform)

    def interpolate_values(self, grid, arr):
        """ Determine interpolated values on a finer grid using the basis functions"""
        # Compute affine transformation per-element to isoparametric element
        xi = grid.J * (grid.arr_fine[1:-1, :] - grid.midpoints[:, None])
        # Legendre polynomials at transformed points
        ps = np.array([sp.legendre(s)(xi) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.transpose(np.tensordot(self.vandermonde_inverse, ps, axes=([0], [0])), [1, 0, 2])
        # Compute interpolated values
        return np.multiply(ell, arr[:, :, None]).sum(axis=1)


def sparse_tensors(tensor):
    # set counter
    counter = 0

    # initialize COO arrays
    non_zeros = []
    row_idx = []
    col_idx = []
    values = []

    # scan array for non-zero entries
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                if tensor[i, j, k] != 0:
                    # update counter
                    counter += 1

                    # set idxs
                    row_idx.append(j)
                    col_idx.append(k)
                    values.append(tensor[i, j, k])

        # set number of non-zeros per row
        non_zeros.append(counter)

    # return as cupy arrays
    return cp.array(non_zeros), cp.array(row_idx), cp.array(col_idx), cp.array(values)


class Basis2D:
    def __init__(self, orders):
        # Build 1D bases
        self.basis_x = Basis1D(orders[0])
        self.basis_y = Basis1D(orders[0])
